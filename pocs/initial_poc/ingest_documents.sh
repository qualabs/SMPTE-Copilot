#!/bin/bash

# Ingest sample documents with appropriate access tags

# Public company overview - accessible to all
docker-compose run --build --rm ingest /app/data/public_company_overview.pdf --tags "Public,General"

# Finance Q4 report - Finance team only
# docker-compose run --rm ingest /app/data/finance_q4_report.pdf --tags "Finance,Confidential,Internal"

# # HR employee handbook - HR and Internal access
# docker-compose run --rm ingest /app/data/hr_employee_handbook.pdf --tags "HR,Internal,Compliance"

# # Technical API documentation - Technical and Public
# docker-compose run --rm ingest /app/data/technical_api_documentation.pdf --tags "Technical,Public"

# # Design system guidelines - Design and Technical teams
# docker-compose run --rm ingest /app/data/design_system_guidelines.pdf --tags "Technical,Internal"

# # Executive strategy plan - Executive role required
# docker-compose run --rm ingest /app/data/executive_strategy_plan.pdf --required-role "Executive" --tags "Strategy,Confidential"

echo "✓ All documents ingested successfully"
