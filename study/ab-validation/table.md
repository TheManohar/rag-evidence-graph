| # | class | query (verbatim) | mode / top_k / filters | A n | B n | overlap | A-only | B-only | B status | B freshness | B grounded/total | traceable citations | A ms | B ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q1 | candidate_facts | `Python backend API development` | hybrid / 5 / - | 5 | 5 | 5 | 0 | 0 | ok | fresh | 5/5 | 5/5 | 39.25 | 74.32 |
| Q2 | candidate_facts | `SQL ETL data pipeline experience` | hybrid / 5 / - | 5 | 5 | 5 | 0 | 0 | ok | fresh | 5/5 | 5/5 | 11.9 | 36.25 |
| Q3 | job_requirements_skills | `Kafka event-driven systems` | hybrid / 5 / - | 5 | 5 | 5 | 0 | 0 | ok | fresh | 5/5 | 5/5 | 11.62 | 37.04 |
| Q4 | job_requirements_skills | `Terraform infrastructure as code` | hybrid / 5 / - | 5 | 5 | 5 | 0 | 0 | ok | fresh | 5/5 | 5/5 | 10.98 | 35.88 |
| Q5 | company_role_lookup | `employer_a senior full-stack engineer python django` | hybrid / 5 / - | 5 | 5 | 5 | 0 | 0 | ok | fresh | 5/5 | 5/5 | 10.83 | 40.19 |
| Q6 | company_role_lookup | `employer_b senior python engineer berlin` | hybrid / 5 / - | 5 | 5 | 5 | 0 | 0 | ok | fresh | 5/5 | 5/5 | 10.6 | 40.49 |
| Q7 | known_reference | `Hermes MCP connection test` | hybrid / 5 / - | 5 | 5 | 5 | 0 | 0 | ok | fresh | 5/5 | 5/5 | 14.92 | 43.47 |
| Q8 | filtered_requirements | `Berlin` | hybrid / 5 / {"types": ["job_opening"]} | 5 | 5 | 5 | 0 | 0 | ok | fresh | 5/5 | 5/5 | 13.98 | 42.9 |
| Q9 | empty_result | `zzqqx flurblewump 99213 nonexistent` | keyword / 5 / - | 0 | 0 | 0 | 0 | 0 | no_evidence | fresh | 0/0 | 0/0 | 10.63 | 38.23 |
| Q10 | empty_result_hybrid_control | `zzqqx flurblewump 99213 nonexistent` | hybrid / 5 / - | 5 | 5 | 5 | 0 | 0 | ok | fresh | 5/5 | 5/5 | 10.31 | 47.82 |
| Q11 | coverage_cap | `Python` | hybrid / 20 / - | 15 | 10 | 10 | 5 | 0 | ok | fresh | 10/15 | 10/10 | 9.78 | 31.25 |
