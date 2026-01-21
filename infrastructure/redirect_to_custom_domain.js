function handler(event) {
    var request = event.request;
    var host = request.headers.host.value;
        
    // If accessing via CloudFront domain, redirect to custom domain
    if (host.endsWith('.cloudfront.net')) {
        var newUrl = 'https://${custom_domain_name}' + request.uri;
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
