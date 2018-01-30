# Prediction_Repeated_Buyers_Double11

Merchants sometimes run big promotions (e.g., discounts or cash coupons) on particular dates (e.g., Boxing-day Sales, "Black Friday" or "Double 11 (Nov 11th)”, in order to attract a large number of new buyers. Unfortunately, many of the attracted buyers are one-time deal hunters, and these promotions may have little long lasting impact on sales. What's more, Tmall.com as the creator of Chinese shopping carnival "Double 11 (Nov 11th)” is threatening by other e-commercial companies like Jingdong, Suning, which resluts in an increasingly high customer churn rate. As more and more customers involving in this shopping festival and more and more competitions appearing in the market, Tmall.com has to reinforce user loyalty to avoid customer loss. 

It is well known that in the field of online advertising, customer targeting is extremely challenging, especially for fresh buyers. However, with the long-term user behavior log accumulated by Tmall.com, we may be able to solve this problem using Machine learning models.

### Business problem

As more and more customers involving in this shopping festival (double 11 promotion) and more and more competitions appearing in the market, Tmall.com has to reinforce user loyalty to avoid customer loss. Both the merchants and Tmall.com have make great efforts on the promotion day. The merchants offer a great discount on all the products to attract new clients on that special day. But the most important is to make them become repeated customers so as to increase the profit in a long term. But the problem is after the big promotion, not many new customers would stay and be transfered to repeated buyers. To alleviate this problem, it is important for merchants to provide personalized service and promotion to different customers. 

On the one hand, it's necessary to identify early who is at risk to leave so as to segment the campaigns to re-engage these clients effectively and in time because it is more expensive and time-consuming acquiring new customers than retaining old ones. This approach is significant for e-commerce companies. Getting new customers always demand a lot of marketing efforts, the merchant has to show trustworthy to potential consumers before they think of clicking the “buy it” button. It generally takes time. Old customers are familiar with the store, they have all the convenience of coming back there without the need of registering on the website or find the store among millions stores since they have already gone through it on their first purchase.

On the other hand, It's also important to identify who can be converted into repeated buyers. By targeting on these potential loyal customers, merchants can greatly reduce the promotion cost and enhance the return on investment (ROI).
By analysing and modeling on customers' behavior dataset, Tmall.com aims to segment potential repeated buyers and buyers who are going to leave in order to increase user loyalty and prevent customer churn.

### Data description

The data set contains anonymized users' shopping logs in the past 6 months before and on the "Double 11" day, and the label information indicating whether they are repeated buyers. Due to privacy issue, data is sampled in a biased way, so the statistical result on this data set would deviate from the actual of Tmall.com. But it will not affect the applicability of the solution. The files for the training and testing data sets can be found in "data_format2.zip". Details of the data format can be found in the table below. We resample the data from 'data_format2.zip' as 'train.csv' and 'test.csv'.Then apply it in our baseline model. Based on these two '.csv' files, we do some transformation and get 'train.csv' and 'test.csv' in order to obtain the better analysis of data in the section of data visulization.

Data are downloaded from a Chiness Kaggle website.
Source : https://tianchi.aliyun.com/getStart/information.htm?spm=5176.100067.5678.2.274d6bb1eahGbL&raceId=231576

### Implementation

Go to [`ramp-worflow`](https://github.com/paris-saclay-cds/ramp-workflow) for more help on the [RAMP](http:www.ramp.studio) ecosystem.

Install ramp-workflow (rampwf), then execute

```
ramp_test_submission
```

to test the starting kit submission (`submissions/starting_kit`) and

```
ramp_test_submission --submission=starting_kit
```

to test `starting_kit` or any other submission in `submissions`.




