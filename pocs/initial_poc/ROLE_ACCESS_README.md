# Role-Aware Access Control

This POC implements a **hybrid tags + roles** approach for document access control using filter-based retrieval.

## Quick Start

### 1. Ingest Documents with Access Control

```bash
# Public document (no restrictions)
docker-compose run --rm ingest python ingest.py /app/data/public.pdf

# Document with access tags
docker-compose run --rm ingest python ingest.py /app/data/financial.pdf --tags "Finance,Internal"

# Document with strict role requirement
docker-compose run --rm ingest python ingest.py /app/data/executive.pdf --required-role "Executive"

# Document with both (hybrid approach)
docker-compose run --rm ingest python ingest.py /app/data/hr_policy.pdf \
  --tags "HR,Compliance" --required-role "HR_Manager"
```

### 2. Query with User Permissions

```bash
# Query as specific role (uses role_mapping.json)
docker-compose run --rm query python query.py "What are the HR policies?" --role "HR_Manager"

# Query with direct tag access
docker-compose run --rm query python query.py "Financial summary?" --tags "Finance,Public"

# Query with both role and tags
docker-compose run --rm query python query.py "Company policies?" --role "Manager" --tags "Special_Access"

# Query without filtering (testing/admin mode)
docker-compose run --rm query python query.py "Anything here?"
```

## Access Control Model

### Metadata Fields

Documents are tagged with two optional fields during ingestion:

- **`access_tags`**: List of tags (e.g., `["Finance", "Public", "Internal"]`)
- **`required_role_strict`**: Single required role (e.g., `"Admin"`)

### Access Decision Logic

A user can access a document if **either** condition is met:

```
(user_role == document.required_role_strict) OR 
(user_authorized_tags ∩ document.access_tags ≠ ∅)
```

Where:
- `user_authorized_tags` = user's direct tags + tags from role mapping

### Filter-Based Retrieval (Strategy 1)

This implementation uses **secure filter-based retrieval**:
- Documents are filtered at the database level (ChromaDB)
- Users only see documents they're authorized to access
- Unauthorized documents are completely hidden from results
- The system acts as if restricted documents don't exist

## Role-to-Tags Mapping (Optional)

Create `/app/role_mapping.json` to map roles to sets of authorized tags:

```json
{
  "Finance_Manager": ["Finance", "Internal", "Reports"],
  "HR_Manager": ["HR", "Internal", "Compliance"],
  "Executive": ["Finance", "HR", "Strategy", "Confidential"],
  "Engineer": ["Technical", "Internal", "Public"],
  "Employee": ["Public", "General"]
}
```

The mapping file is optional but recommended for easier role management.

## Examples

### Example 1: Finance Document

Ingest a financial report:
```bash
docker-compose run --rm ingest python ingest.py /app/data/q4_report.pdf \
  --tags "Finance,Confidential,Q4_2024"
```

Query as Finance Manager (has "Finance" tag via role mapping):
```bash
docker-compose run --rm query python query.py "Q4 revenue?" --role "Finance_Manager"
# ✓ Can access the document
```

Query as Engineer (doesn't have "Finance" tag):
```bash
docker-compose run --rm query python query.py "Q4 revenue?" --role "Engineer"
# ✗ Document filtered out, no results returned
```

### Example 2: Executive-Only Document

Ingest an executive memo:
```bash
docker-compose run --rm ingest python ingest.py /app/data/exec_memo.pdf \
  --required-role "Executive"
```

Query as Executive (exact role match):
```bash
docker-compose run --rm query python query.py "Executive decisions?" --role "Executive"
# ✓ Can access the document
```

Query as Manager (no role match, no tag access):
```bash
docker-compose run --rm query python query.py "Executive decisions?" --role "Manager"
# ✗ Document filtered out, no results returned
```

### Example 3: Hybrid Access

Ingest HR policy with hybrid access:
```bash
docker-compose run --rm ingest python ingest.py /app/data/hr_policy.pdf \
  --tags "HR,Compliance" --required-role "HR_Manager"
```

Who can access:
- HR_Manager (strict role match) ✓
- Anyone with "HR" or "Compliance" tags ✓
- Executives (if they have "HR" tag in role mapping) ✓
- Engineers (no role match, no tag access) ✗

## Security Notes

- **Database-Level Filtering**: Access control is enforced by ChromaDB filters, not by the LLM
- **Least Privilege**: Users only see what they're authorized to see
- **No Hints**: Unauthorized documents are completely hidden (no "access denied" messages)
- **No Restrictions = Public**: Documents without tags/roles are accessible to all users
- **Testing Mode**: Omit `--role` and `--tags` to bypass filtering (for testing/admin access)

## Implementation Details

### Ingestion Flow
1. Parse PDF to Markdown
2. Chunk Markdown into smaller pieces
3. **Add access metadata** to each chunk (tags, required_role_strict)
4. Embed chunks with access metadata
5. Store in ChromaDB with metadata

### Query Flow
1. Parse user permissions (role + direct tags)
2. Load role mapping (if available)
3. Build authorized tag set (direct tags + role-mapped tags)
4. **Build ChromaDB filter** (role match OR tag intersection)
5. Apply filter during vector search
6. Return only accessible documents

## Troubleshooting

### No results returned
- Check if documents were ingested with access tags
- Verify user role/tags match document access requirements
- Try querying without filters to test if documents exist

### Role mapping not working
- Ensure `role_mapping.json` is mounted in docker-compose.yml
- Check JSON syntax is valid
- Verify role names match exactly (case-sensitive)

### Access control not applied
- Ensure you're passing `--role` or `--tags` arguments to query.py
- Without these arguments, no filtering is applied (admin mode)
