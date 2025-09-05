"""
SQL Queries for Superset Data Extraction

This module contains predefined SQL queries that can be used with the SupersetExtractor.
"""

# 9/5/2025 Fake Data Analysis
SQL_FAKE_DATA_PARTY = """
SELECT
   opportunity_uservisit.opportunity_id AS opportunity_id,
   opportunity_uservisit.user_id AS flw_id,
   users_user.name AS flw_name,
   opportunity_uservisit.visit_date,
   form_json
   
FROM opportunity_uservisit
LEFT JOIN opportunity_opportunity
 ON opportunity_opportunity.id = opportunity_uservisit.opportunity_id
LEFT JOIN users_user
 ON opportunity_uservisit.user_id = users_user.id
WHERE opportunity_opportunity.id IN (716,715);
"""

SQL_ALL_DATA_QUERY = """
SELECT
   opportunity_uservisit.opportunity_id AS opportunity_id,
   opportunity_uservisit.user_id AS flw_id,
   form_json
   
FROM opportunity_uservisit
LEFT JOIN opportunity_opportunity
 ON opportunity_opportunity.id = opportunity_uservisit.opportunity_id
WHERE opportunity_opportunity.id IN (601, 575, 597, 531, 412, 516, 598, 604, 573, 595, 566, 539, 411, 579, 517);
"""
# End Fake Data Analysis