# Credit_Card_Lead_Prediction-JOB-A-THON
Participated in Analytics Vidya Hackathon ( JOB-A-THON | May 2021 ). Public roc_auc_score - `0.8730946`.

This Repository contains all code, reports and approach.


## Project Details:

Happy Customer Bank is a mid-sized private bank that deals in all kinds of banking products, like Savings accounts,
Current accounts, investment products, credit products, among other offerings.
The bank also cross-sells products to its existing customers and to do so they use different kinds of communication like
tele-calling, e-mails, recommendations on net banking, mobile banking, etc.
In this case, the Happy Customer Bank wants to cross sell its credit cards to its existing customers. The bank has identified
a set of customers that are eligible for taking these credit cards.
Now, the bank is looking for your help in identifying customers that could show higher intent towards a recommended credit
card, given:
* Customer details (gender, age, region etc.)
* Details of his/her relationship with the bank (Channel_Code,Vintage, 'Avg_Asset_Value etc.)


## Dataset Description:

* `train.csv` - ID,Gender,Age,Region_Code,Occupation,Channel_code,Vintage,Credit_Product,Avg_Account_balance,Is_Active,Is_Lead
* `test.csv` - ID,Gender,Age,Region_Code,Occupation,Channel_code,Vintage,Credit_Product,Avg_Account_balance,Is_Active
* `sample_submission` - ID,Is_Lead


## Tools

**Code:** Python Version: 3.8

**For data wrangling and visualization:** scikit-learn , Pandas Profiling ,SciPy

**For predictive analytics:** scikit-learn, LightGBM, Catboost 

**For Reporting:** Google Slides

## Task 

Build a classifier that predicts if the customer is a interested in credit card or not.


## Report 

Credit Card Lead Prediction Report
