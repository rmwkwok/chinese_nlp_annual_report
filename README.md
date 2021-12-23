## Objective
Identify named-entity (organization and person) from company annual report in Chinese, and their person-person and person-organization relations.

This is a baseline work.

## Work flow, precision, and actionable discussed

![](https://raw.githubusercontent.com/rmwkwok/chinese_nlp_annual_report/main/report.png)

# Numbers

1. NER model discovered 1063 persons / 2842 organizations

2. Simple rule-based filtered to 549 persons / 1870 organizations (<50%, 66% precisions respectively)

3. Among the 549 persons, precision estimated to be 52%; among the organizations, 45%

4. 890 pairs of person-to-person or person-to-org discovered

5. 221 pairs of relations discovered

6. Precision among Family relations is 60%

## NER filter rules

1. Keep only person and organization

2. Person name length: English >= 6, Chinese 2-4; Organization name length: English >=6, Chinese >=4

3. Person names not in ['於集團擔', '附註'] which are likely to be training data-biased

## Relation extraction filter rules

1. Keep only person-to-person and person-to-organization pairs

2. Remove relation identified as 'Unknown'

3. Remove relation other than 'Work' for person-to-organization

## Tech

1. NER Model (WC-LSTM) trained with Chinese Resume dataset ([Credit](https://github.com/zerohd4869/Chinese-NER)). Resume data is chosen for similarity to personal introductions in annual reports.

2. Relation extraction model (Bert) trained with a dataset of various types of interpersonal relationships ([Credit](https://github.com/Jacen789/relation-extraction)). It was chosen to mainly discover interpersonal relations. 

3. Python 3.8.10, pytorch 1.10.1+cpu
