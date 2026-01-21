# ACM Certificate for Custom Domain
resource "aws_acm_certificate" "cloudfront" {
  provider                  = aws.us_east_1
  domain_name               = var.custom_domain_name
  validation_method         = "DNS"

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name = "${var.project_name}-cloudfront-cert"
  }
}

# ACM Certificate Validation (only created when custom domain is enabled)
resource "aws_acm_certificate_validation" "cloudfront" {
  count                   = var.enable_custom_domain ? 1 : 0
  provider                = aws.us_east_1
  certificate_arn         = aws_acm_certificate.cloudfront.arn
  validation_record_fqdns = [for record in aws_acm_certificate.cloudfront.domain_validation_options : record.resource_record_name]
}

# CloudFront Managed Policies
data "aws_cloudfront_cache_policy" "caching_disabled" {
  name = "Managed-CachingDisabled"
}

data "aws_cloudfront_origin_request_policy" "all_viewer" {
  name = "Managed-AllViewer"
}

# CloudFront Prefix List for Origin-Facing IPs
data "aws_ec2_managed_prefix_list" "cloudfront" {
  name = "com.amazonaws.global.cloudfront.origin-facing"
}

# CloudFront Function to redirect default domain to custom domain
resource "aws_cloudfront_function" "redirect_to_custom_domain" {
  count   = var.enable_custom_domain ? 1 : 0
  name    = "${var.project_name}-redirect-to-custom-domain"
  runtime = "cloudfront-js-1.0"
  comment = "Redirect CloudFront default domain to custom domain"
  publish = true
  code    = <<-EOT
    function handler(event) {
        var request = event.request;
        var host = request.headers.host.value;
        
        // If accessing via CloudFront domain, redirect to custom domain
        if (host.endsWith('.cloudfront.net')) {
            var newUrl = 'https://${var.custom_domain_name}' + request.uri;
            if (request.querystring && request.querystring.value) {
                newUrl += '?' + request.querystring.value;
            }
            return {
                statusCode: 301,
                statusDescription: 'Moved Permanently',
                headers: {
                    'location': { value: newUrl }
                }
            };
        }
        
        return request;
    }
  EOT
}

# CloudFront Distribution for OpenWebUI
resource "aws_cloudfront_distribution" "openwebui" {
  enabled         = true
  is_ipv6_enabled = true
  comment         = "CloudFront distribution for OpenWebUI"
  aliases         = var.enable_custom_domain ? [var.custom_domain_name] : []

  origin {
    domain_name = aws_eip.server.public_dns
    origin_id   = "openwebui-ec2-origin"

    custom_origin_config {
      http_port              = 3000
      https_port             = 443
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    allowed_methods        = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods         = ["GET", "HEAD", "OPTIONS"]
    target_origin_id       = "openwebui-ec2-origin"
    viewer_protocol_policy = "https-only"

    cache_policy_id          = data.aws_cloudfront_cache_policy.caching_disabled.id
    origin_request_policy_id = data.aws_cloudfront_origin_request_policy.all_viewer.id

    # Redirect CloudFront domain to custom domain when enabled
    dynamic "function_association" {
      for_each = var.enable_custom_domain ? [1] : []
      content {
        event_type   = "viewer-request"
        function_arn = aws_cloudfront_function.redirect_to_custom_domain[0].arn
      }
    }
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  # Conditional viewer certificate configuration
  viewer_certificate {
    cloudfront_default_certificate = var.enable_custom_domain ? false : true
    acm_certificate_arn            = var.enable_custom_domain ? aws_acm_certificate.cloudfront.arn : null
    ssl_support_method             = var.enable_custom_domain ? "sni-only" : null
    minimum_protocol_version       = var.enable_custom_domain ? "TLSv1.2_2021" : null
  }

  tags = {
    Name = "${var.project_name}-openwebui-cdn"
  }
}
