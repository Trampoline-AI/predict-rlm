# Sample Output — Document Redaction

**Input:** [PNFS-Employment-Agreement-2025.pdf](../input/PNFS-Employment-Agreement-2025.pdf) (6 pages)
**Redacted output:** [PNFS-Employment-Agreement-2025_redacted.pdf](PNFS-Employment-Agreement-2025_redacted.pdf)

## Summary

Total redactions applied: **89**

## Per-Page Summary

| Page | Redactions | Categories |
|---:|---:|---|
| 0 | 14 | address, date, email, government_id, person_name, phone_number |
| 1 | 13 | address, date, financial_info, government_id, person_name |
| 2 | 26 | address, date, email, government_id, person_name, phone_number |
| 3 | 13 | date, email, financial_info, other_id, person_name, phone_number |
| 4 | 18 | address, date, email, government_id, person_name, phone_number |
| 5 | 5 | address, date, government_id, person_name |

## Redaction Targets

### Page 0 — Employment Agreement (cover)

| Category | Text | Reason |
|---|---|---|
| address | 1847 Harbour View Drive, Suite 400 | Street address must be redacted |
| address | Vancouver, BC V6E 3S7 | Part of mailing address including postal code |
| date | Date: March 15, 2025 | Contains a specific date |
| person_name | Margaret Elisabeth Thornbury-Watson | Full name of an individual employee |
| date | Date of Birth: September 12, 1987 | Contains a birthdate |
| government_id | Social Insurance Number: 847-291-036 | Social insurance/government identification number |
| address | Home Address: 2934 Cypress Crescent, North Vancouver, BC V7R 2T8 | Full home street address |
| email | m.thornbury.watson@gmail.com | Personal email address |
| phone_number | Mobile Phone: (604) 889-3247 | Personal mobile phone number |
| person_name | David Watson | Full name of emergency contact individual |
| phone_number | (604) 773-5518 | Emergency contact phone number |
| date | Start Date: April 1, 2025 | Contains an employment start date |
| person_name | James Harrington | Full name of reporting manager |
| person_name | Robert Chen | Full name of employer representative |

### Page 1 — Employee Details and Compensation

| Category | Text | Reason |
|---|---|---|
| person_name | Margaret Elisabeth Thornbury-Watson | Full legal name of an individual |
| person_name | Margaret | Preferred first name of an individual |
| government_id | QK894217 | Passport number |
| government_id | 7284913 | Driver's license number |
| address | 2934 Cypress Crescent | Street address |
| address | North Vancouver, BC V7R 2T8 | City with province and postal code |
| address | Canada | Country line as part of full mailing address |
| financial_info | 5127849 | Bank account number |
| person_name | Margaret E. Thornbury-Watson | Account holder full name |
| person_name | David Watson | Life insurance beneficiary name |
| date | June 3, 1985 | Beneficiary date of birth |
| government_id | 912-347-058 | Social insurance number |
| financial_info | 4891-7723-0056 | RRSP account number |

### Page 2 — References and Background Verification

| Category | Text | Reason |
|---|---|---|
| person_name | Dr. Patricia Holmgren | Full name of an individual |
| person_name | Alistair McKinnon | Full name of an individual |
| person_name | Samantha Reeves-Park | Full name of an individual |
| person_name | Margaret Elisabeth Thornbury-Watson | Full name of an individual |
| person_name | Margaret E. Thornbury-Watson | Full name of an individual |
| phone_number | Phone: (416) 307-8842 | Contains a phone number |
| phone_number | Phone: (604) 691-3200 ext. 4417 | Contains a phone number |
| phone_number | Phone: (250) 412-9933 | Contains a phone number |
| email | patricia.holmgren@td.com | Email address |
| email | amckinnon@kpmg.ca | Email address |
| email | s.reeves.park@cascadiarenewable.ca | Email address |
| government_id | Social Insurance Number (for credit check): 847-291-036 | Contains Social Insurance Number |
| date | Date of Birth (for identity verification): September 12, 1987 | Contains date of birth |
| address | 66 Wellington Street West, Toronto, ON M5K 1A2 | Full street address including postal code |
| date | Dates of Employment: January 2019 - November 2023 | Employment dates |
| date | Date: March 15, 2025 | Authorization date |

### Page 3 — IT Onboarding and System Access

| Category | Text | Reason |
|---|---|---|
| email | m.thornbury-watson@pnfs.ca | Email address that identifies an individual |
| other_id | EMP-2025-0847 | Unique employee identifier |
| other_id | YVR-04-2291 | Unique badge identifier |
| other_id | 4NQ9X83-J72M | Unique hardware serial number |
| phone_number | (604) 812-0094 | Phone number |
| other_id | 353847291047823 | Unique IMEI hardware identifier |
| other_id | P-2025-0312 | Unique parking permit identifier |
| other_id | HVT-8847-MW | Unique building access identifier |
| financial_info | 4519-8847-2291-0036 | Full payment card number |
| person_name | Margaret Thornbury-Watson | Full name of an individual |
| date | 03/2028 | Represents an expiry date |
| person_name | James Harrington | Full name of an individual |

### Page 4 — Health, Safety, and Family Information

| Category | Text | Reason |
|---|---|---|
| person_name | David Watson | Full name of individual |
| person_name | Eleanor Thornbury | Full name of individual |
| person_name | David James Watson | Full name of individual |
| person_name | Oliver Thornbury Watson | Full name of individual |
| person_name | Clara Thornbury Watson | Full name of individual |
| phone_number | (604) 773-5518 | Phone number |
| phone_number | (250) 884-6617 | Phone number |
| email | david.j.watson@outlook.com | Email address |
| address | 2934 Cypress Crescent, North Vancouver, BC V7R 2T8 | Street address |
| address | 445 Linden Avenue, Victoria, BC V8V 4G5 | Street address |
| government_id | 9847 291 036 | Health card number |
| date | June 3, 1985 | Full date of birth |
| date | November 22, 2019 | Full date of birth |
| date | August 8, 2022 | Full date of birth |
| government_id | 9912 347 058 | Health card number |
| government_id | 9201 947 223 | Health card number |
| government_id | 9208 822 491 | Health card number |
| government_id | 912-347-058 | Social Insurance Number |

### Page 5 — Non-Disclosure and Non-Compete

| Category | Text | Reason |
|---|---|---|
| person_name | Margaret Elisabeth Thornbury-Watson | Full name of an individual |
| government_id | 847-291-036 | SIN is a government identification number |
| address | 2934 Cypress Crescent, North Vancouver, BC V7R 2T8 | Full street mailing address |
| date | March 15, 2025 | Specific calendar date |
| person_name | Robert Chen | Full name of an individual |

## Run Stats

| | Main LM | Sub-LM |
|---|---|---|
| Model | `openai/gpt-5.4` | `openai/gpt-5.1` |
| Calls | 4 | 30 |
| Input tokens | 28,255 | 27,944 |
| Output tokens | 3,144 | 6,504 |
| Cost | $0.08 | $0.10 |

| Metric | Value |
|---|---|
| Document | 1 file, 6 pages |
| Duration | 1m 27s |
| Total cost | $0.18 |
| Cost per page | $0.03 |
