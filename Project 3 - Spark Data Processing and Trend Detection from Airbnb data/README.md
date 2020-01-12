## Project 3: Spark Data Processing and Trend Detection from Airbnb data

### _by Sebastian Sbirna (s190553), Yingrui Li (s171353) and Aijie Shu (s182190)_
---
**This is the third of three mandatory projects to be handed in as part of the assessment for the course 02807 Computational Tools for Data Science at Technical University of Denmark, autumn 2019.**

# Introduction
[Airbnb](http://airbnb.com) is an online marketplace for arranging or offering lodgings. 

___In this project you will use Spark to analyze data obtained from the Airbnb website. The purpose of the analysis is to extract information about trends and patterns from the data.___

The project has two parts.

### Part 1: Loading, describing and preparing the data
There's quite a lot of data. Make sure that you can load and correctly parse the data, and that you understand what the dataset contains. You should also prepare the data for the analysis in part two. This means cleaning it and staging it so that subsequent queries are fast.

### Par 2: Analysis
In this part your goal is to learn about trends and usage patterns from the data. You should give solutions to the tasks defined in this notebook, and you should use Spark to do the data processing. You may use other libraries like for instance Pandas and matplotlib for visualisation.

## Guidelines
- Processing data should be done using Spark. Once data has been reduced to aggregate form, you may use collect to extract it into Python for visualisation.
- Your solutions will be evaluated by correctness, code quality and interpretability of the output. This means that you have to write clean and efficient Spark code that will generate sensible execution plans, and that the tables and visualisations that you produce are meaningful and easy to read.
- You may add more cells for your solutions, but you should not modify the notebook otherwise.

### Create Spark session and define imports


```python
from pyspark.sql import *
from pyspark.sql import functions as f
from pyspark.sql.types import *
from pyspark.sql.functions import split, explode
```


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from bokeh.io import output_notebook
from bokeh.io import show
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool
from bokeh.plotting import figure
import string
```


```python
# !jupyter nbextension enable --py --sys-prefix widgetsnbextension
```


```python
spark = SparkSession.builder.appName("Project3").getOrCreate()
```

# Part 1: Loading, describing and preparing the data
The data comes in two files. Start by downloading the files and putting them in your `data/` folder.

- [Listings](https://files.dtu.dk/u/siPzAasj8w2gI_ME/listings.csv?l) (5 GB)
- [Reviews](https://files.dtu.dk/u/k3oaPYp6GjKBeho4/reviews.csv?l) (9.5 GB)

### Load the data
The data has multiline rows (rows that span multiple lines in the file). To correctly parse these you should use the `multiline` option and set the `escape` character to be `"`.


```python
path1 = '/data/listings.csv'
path2 = '/data/reviews.csv'
```


```python
listings = spark.read\
  .option("multiLine", True)\
  .option("header", True)\
  .option("escape", "\"")\
  .option("inferSchema", True)\
  .csv(path1)
```


```python
reviews = spark.read\
  .option("multiLine", True)\
  .option("header", True)\
  .option("escape", "\"")\
  .option("inferSchema", True)\
  .csv(path2)
```

### Describe the data
List the features (schema) and sizes of the datasets.

##### Listings


```python
listings.printSchema()    
```

    root
     |-- id: string (nullable = true)
     |-- listing_url: string (nullable = true)
     |-- scrape_id: string (nullable = true)
     |-- last_scraped: string (nullable = true)
     |-- name: string (nullable = true)
     |-- summary: string (nullable = true)
     |-- space: string (nullable = true)
     |-- description: string (nullable = true)
     |-- experiences_offered: string (nullable = true)
     |-- neighborhood_overview: string (nullable = true)
     |-- notes: string (nullable = true)
     |-- transit: string (nullable = true)
     |-- access: string (nullable = true)
     |-- interaction: string (nullable = true)
     |-- house_rules: string (nullable = true)
     |-- thumbnail_url: string (nullable = true)
     |-- medium_url: string (nullable = true)
     |-- picture_url: string (nullable = true)
     |-- xl_picture_url: string (nullable = true)
     |-- host_id: string (nullable = true)
     |-- host_url: string (nullable = true)
     |-- host_name: string (nullable = true)
     |-- host_since: string (nullable = true)
     |-- host_location: string (nullable = true)
     |-- host_about: string (nullable = true)
     |-- host_response_time: string (nullable = true)
     |-- host_response_rate: string (nullable = true)
     |-- host_acceptance_rate: string (nullable = true)
     |-- host_is_superhost: string (nullable = true)
     |-- host_thumbnail_url: string (nullable = true)
     |-- host_picture_url: string (nullable = true)
     |-- host_neighbourhood: string (nullable = true)
     |-- host_listings_count: string (nullable = true)
     |-- host_total_listings_count: string (nullable = true)
     |-- host_verifications: string (nullable = true)
     |-- host_has_profile_pic: string (nullable = true)
     |-- host_identity_verified: string (nullable = true)
     |-- street: string (nullable = true)
     |-- neighbourhood: string (nullable = true)
     |-- neighbourhood_cleansed: string (nullable = true)
     |-- neighbourhood_group_cleansed: string (nullable = true)
     |-- city: string (nullable = true)
     |-- state: string (nullable = true)
     |-- zipcode: string (nullable = true)
     |-- market: string (nullable = true)
     |-- smart_location: string (nullable = true)
     |-- country_code: string (nullable = true)
     |-- country: string (nullable = true)
     |-- latitude: string (nullable = true)
     |-- longitude: string (nullable = true)
     |-- is_location_exact: string (nullable = true)
     |-- property_type: string (nullable = true)
     |-- room_type: string (nullable = true)
     |-- accommodates: string (nullable = true)
     |-- bathrooms: string (nullable = true)
     |-- bedrooms: string (nullable = true)
     |-- beds: string (nullable = true)
     |-- bed_type: string (nullable = true)
     |-- amenities: string (nullable = true)
     |-- square_feet: string (nullable = true)
     |-- price: string (nullable = true)
     |-- weekly_price: string (nullable = true)
     |-- monthly_price: string (nullable = true)
     |-- security_deposit: string (nullable = true)
     |-- cleaning_fee: string (nullable = true)
     |-- guests_included: string (nullable = true)
     |-- extra_people: string (nullable = true)
     |-- minimum_nights: string (nullable = true)
     |-- maximum_nights: string (nullable = true)
     |-- minimum_minimum_nights: string (nullable = true)
     |-- maximum_minimum_nights: string (nullable = true)
     |-- minimum_maximum_nights: string (nullable = true)
     |-- maximum_maximum_nights: string (nullable = true)
     |-- minimum_nights_avg_ntm: string (nullable = true)
     |-- maximum_nights_avg_ntm: string (nullable = true)
     |-- calendar_updated: string (nullable = true)
     |-- has_availability: string (nullable = true)
     |-- availability_30: string (nullable = true)
     |-- availability_60: string (nullable = true)
     |-- availability_90: string (nullable = true)
     |-- availability_365: string (nullable = true)
     |-- calendar_last_scraped: string (nullable = true)
     |-- number_of_reviews: string (nullable = true)
     |-- number_of_reviews_ltm: string (nullable = true)
     |-- first_review: string (nullable = true)
     |-- last_review: string (nullable = true)
     |-- review_scores_rating: string (nullable = true)
     |-- review_scores_accuracy: string (nullable = true)
     |-- review_scores_cleanliness: string (nullable = true)
     |-- review_scores_checkin: string (nullable = true)
     |-- review_scores_communication: string (nullable = true)
     |-- review_scores_location: string (nullable = true)
     |-- review_scores_value: string (nullable = true)
     |-- requires_license: string (nullable = true)
     |-- license: string (nullable = true)
     |-- jurisdiction_names: string (nullable = true)
     |-- instant_bookable: string (nullable = true)
     |-- is_business_travel_ready: string (nullable = true)
     |-- cancellation_policy: string (nullable = true)
     |-- require_guest_profile_picture: string (nullable = true)
     |-- require_guest_phone_verification: string (nullable = true)
     |-- calculated_host_listings_count: string (nullable = true)
     |-- calculated_host_listings_count_entire_homes: string (nullable = true)
     |-- calculated_host_listings_count_private_rooms: string (nullable = true)
     |-- calculated_host_listings_count_shared_rooms: string (nullable = true)
     |-- reviews_per_month: string (nullable = true)
    
    


```python
listings.count()
```




    1330480



##### Reviews


```python
reviews.printSchema()
```

    root
     |-- listing_id: string (nullable = true)
     |-- id: string (nullable = true)
     |-- date: string (nullable = true)
     |-- reviewer_id: string (nullable = true)
     |-- reviewer_name: string (nullable = true)
     |-- comments: string (nullable = true)
    
    


```python
reviews.count()
```




    32297300



### Prepare the data for analysis
You should prepare two dataframes to be used in the analysis part of the project. You should not be concerned with cleaning the data. There's a lot of it, so it will be sufficient to drop rows that have bad values. You may want to go back and refine this step at a later point when doing the analysis.

You may also want to consider if you can stage your data so that subsequent processing is more efficient (this is not strictly necessary for Spark to run, but you may be able to decrease the time you sit around waiting for Spark to finish things)

<font color='blue'>
1. filter out necessary columns which will be involved in following questions.</font>
<font color='blue'>    
2. Keep the `NaN` entries and only drop them when necessary 
</font>


```python
listings_sub = listings.select(f.col('id').alias('listing_id'), 'city', 'neighbourhood', 'property_type', 'price', 'review_scores_rating')\
               .drop_duplicates()
```


```python
listings_sub.show()
```

    +----------+---------------+--------------------+-------------+---------+--------------------+
    |listing_id|           city|       neighbourhood|property_type|    price|review_scores_rating|
    +----------+---------------+--------------------+-------------+---------+--------------------+
    |   1170802|      Stockholm|           Skarpnäck|    Apartment|  $352.00|                null|
    |   1867260|      Stockholm|            Norrmalm|    Apartment|$1,301.00|                 100|
    |   3108437|     Stockholm |Hägersten-Liljeho...|    Apartment|  $196.00|                  73|
    |   3259719|      Stockholm|           Södermalm|    Apartment|$1,996.00|                  97|
    |   5218468|      Stockholm|          Skärholmen|        House|  $949.00|                 100|
    |   5360266|      Stockholm|           Skarpnäck|    Townhouse|$1,771.00|                  97|
    |   5706203|      Stockholm|         Kungsholmen|    Apartment|  $890.00|                  93|
    |   6340350|      Stockholm|           Östermalm|    Apartment|  $998.00|                  96|
    |   7063087|      Stockholm|Hägersten-Liljeho...|    Apartment|  $900.00|                  97|
    |   7614052|      Stockholm|         Kungsholmen|    Apartment|  $695.00|                  91|
    |   7616523|      Stockholm|              Farsta|    Apartment|  $274.00|                  93|
    |  13255147|      Stockholm|         Kungsholmen|    Apartment|$1,154.00|                 100|
    |  14018055|      Stockholm|Hägersten-Liljeho...|    Apartment|  $597.00|                  73|
    |  16190458|      Stockholm|            Norrmalm|    Apartment|$1,800.00|                null|
    |  16654904|      Stockholm|            Norrmalm|    Apartment|  $998.00|                  96|
    |  18290399|      Stockholm|            Norrmalm|    Apartment|$1,497.00|                null|
    |  18840650|      Stockholm|           Skarpnäck|    Apartment|  $949.00|                 100|
    |  19522814|Ladugårdsgärdet|           Östermalm|    Apartment|  $479.00|                  93|
    |  19966118|       Norrmalm|            Norrmalm|    Apartment|$1,096.00|                  88|
    |  20633450|Skarpnäcks Gård|           Skarpnäck|    Apartment|  $597.00|                null|
    +----------+---------------+--------------------+-------------+---------+--------------------+
    only showing top 20 rows
    
    


```python
listings_sub.count()
```




    1330380



---
# Part 2: Analysis
Use Spark and your favorite tool for data visualization to solve the following tasks.

## The basics
### Compute and show a dataframe with the number of listings and neighbourhoods per city.


```python
city_neigh_listing = listings_sub.groupBy('city')\
                     .agg(f.countDistinct('listing_id').alias('listing_counts'), f.countDistinct('neighbourhood').alias('neighbourhood_counts'))\
                     .filter(f.col('city').isNotNull())\
                     .orderBy(f.desc('listing_counts'))
```


```python
city_neigh_listing.show(5)
```

    +--------------+--------------+--------------------+
    |          city|listing_counts|neighbourhood_counts|
    +--------------+--------------+--------------------+
    |         Paris|         61923|                  63|
    |Greater London|         46521|                 149|
    |        London|         33100|                 148|
    |       Beijing|         32338|                  61|
    |   Los Angeles|         27763|                  97|
    +--------------+--------------+--------------------+
    only showing top 5 rows
    
    

Based on the table above, you should choose a city that you want to continue your analysis for. The city should have mulitple neighbourhoods with listings in them.



<font color='blue'> We choose Beijing for furthur analysis. It contains `32338` listings and `61` neighbourhood</font>

### Compute and visualize the number of listings of different property types per neighbourhood in your city.


```python
beijing_neigh_listing = listings_sub.groupBy('city', 'neighbourhood', 'property_type')\
                        .agg(f.countDistinct('listing_id').alias('listing_counts_by_type'))\
                        .filter(f.col('city')=='Beijing')\
                        .filter(f.col('neighbourhood').isNotNull())\
                        .filter(f.col('property_type').isNotNull())\
                        .orderBy(f.desc('listing_counts_by_type'))
```


```python
beijing_neigh_listing.show()
```

    +-------+------------------+------------------+----------------------+
    |   city|     neighbourhood|     property_type|listing_counts_by_type|
    +-------+------------------+------------------+----------------------+
    |Beijing|          Chaoyang|         Apartment|                  3238|
    |Beijing|          Chaoyang|       Condominium|                  1462|
    |Beijing|           Haidian|         Apartment|                   821|
    |Beijing|           Fengtai|         Apartment|                   777|
    |Beijing|          Chaoyang|             House|                   701|
    |Beijing|         Dongcheng|         Apartment|                   636|
    |Beijing|           Haidian|       Condominium|                   436|
    |Beijing|          Chaoyang|Serviced apartment|                   402|
    |Beijing|          Sanlitun|         Apartment|                   378|
    |Beijing|          Chaoyang|              Loft|                   352|
    |Beijing|           Xicheng|         Apartment|                   304|
    |Beijing|           Fengtai|       Condominium|                   291|
    |Beijing|Jinsong/Panjiayuan|         Apartment|                   281|
    |Beijing|           Xicheng|             House|                   279|
    |Beijing|           Haidian|             House|                   271|
    |Beijing|         Dongcheng|             House|                   255|
    |Beijing|         Dongcheng|       Condominium|                   255|
    |Beijing|          Wangjing|         Apartment|                   233|
    |Beijing|           Fengtai|             House|                   221|
    |Beijing|       Chongwenmen|         Apartment|                   190|
    +-------+------------------+------------------+----------------------+
    only showing top 20 rows
    
    


```python
# convert to pandas dataframe for visualization
beijing_df = beijing_neigh_listing.toPandas()
```


```python
# visualize the number of differnt property type in different neighbourhood
plot = sns.FacetGrid(beijing_df, col='neighbourhood', sharex=False, sharey=False, col_wrap=3, size=6)
plot.map(sns.barplot, 'property_type', 'listing_counts_by_type')
plot.set_xticklabels(rotation=90)
plot.fig.tight_layout()
for ax in plot.axes.flat:
    ax.set_ylabel('listing number of each property type')
```

    /anaconda3/lib/python3.6/site-packages/seaborn/axisgrid.py:230: UserWarning: The `size` paramter has been renamed to `height`; please update your code.
      warnings.warn(msg, UserWarning)
    /anaconda3/lib/python3.6/site-packages/seaborn/axisgrid.py:715: UserWarning: Using the barplot function without specifying `order` is likely to produce an incorrect plot.
      warnings.warn(warning)
    


![png](output_33_1.png)


## Prices
### Compute the minimum, maximum and average listing price in your city.

<font color='blue'>    
In this part, we first check the price format and clean/transfer it. Then find the min, max, avg.
</font>


```python
# filter out Beijing
beijing_price = listings_sub.filter(f.col('city')=='Beijing')                         
```


```python
# define a function to get the currency sign of a price
extract_currency = f.udf(lambda x : x[0])   
```


```python
# check if there is only one type of currency sign -- $
beijing_price.withColumn('currency', extract_currency(f.col('price')))\
             .select('currency').distinct()\
             .show()
```

    +--------+
    |currency|
    +--------+
    |       $|
    +--------+
    
    


```python
from re import sub
from decimal import Decimal

# define a function to clean the price. only keep numbers and drop $ and . ,
extract_price = f.udf(lambda price : Decimal(sub(r'[^\d.]', '', price)))
```


```python
# apply the function on column price and make a new column
beijing_price = beijing_price.withColumn('price', extract_price(f.col('price')).cast(DoubleType()))
```


```python
# sort the price
beijing_price.orderBy(f.desc('price')).show()
```

    +----------+-------+--------------------+------------------+-------+--------------------+
    |listing_id|   city|       neighbourhood|     property_type|  price|review_scores_rating|
    +----------+-------+--------------------+------------------+-------+--------------------+
    |  15666822|Beijing|                null|             House|71597.0|                null|
    |  12689987|Beijing|            Chaoyang|         Apartment|71110.0|                 100|
    |  28134193|Beijing|             Haidian|         Apartment|68984.0|                  60|
    |  22382855|Beijing|            Chaoyang|       Condominium|68980.0|                 100|
    |  35658816|Beijing|            Chaoyang|         Apartment|68088.0|                null|
    |  34071825|Beijing|            Chaoyang|             House|68002.0|                null|
    |  27587044|Beijing|                null|       Condominium|66665.0|                null|
    |  28803519|Beijing|           Dongcheng|       Condominium|65973.0|                  80|
    |  15488817|Beijing|              Zhuang|Serviced apartment|63345.0|                  86|
    |  21942314|Beijing|             Fengtai|             House|60002.0|                 100|
    |  20748712|Beijing|            Chaoyang|       Condominium|60002.0|                  60|
    |  29138170|Beijing|            Chaoyang|       Condominium|60002.0|                null|
    |  31432793|Beijing|Liang Ma Qiao/San...|         Apartment|60002.0|                 100|
    |  32905425|Beijing|                null|             House|55003.0|                 100|
    |  35353869|Beijing|                null|       Condominium|46164.0|                  95|
    |  29520327|Beijing|            Chaoyang|             House|43798.0|                null|
    |  15526505|Beijing|             Shilipu|         Apartment|39358.0|                 100|
    |  31543015|Beijing|                null|         Apartment|36771.0|                null|
    |  30355531|Beijing|             Fengtai|             House|30001.0|                null|
    |  25434257|Beijing|                null|            Hostel|28501.0|                null|
    +----------+-------+--------------------+------------------+-------+--------------------+
    only showing top 20 rows
    
    


```python
beijing_price.count()
```




    32338




```python

beijing_price.filter(f.col('neighbourhood').isNull()).count()
```




    11416



<font color='blue'>    
According to above counts, 1/3 of the `neighbourhood` entries are null. So we decide to drop them.
</font>


```python
# drop null value
beijing_price = beijing_price.filter(f.col('neighbourhood').isNotNull())
beijing_price.orderBy(f.desc('price')).show()
```

    +----------+-------+--------------------+------------------+-------+--------------------+
    |listing_id|   city|       neighbourhood|     property_type|  price|review_scores_rating|
    +----------+-------+--------------------+------------------+-------+--------------------+
    |  12689987|Beijing|            Chaoyang|         Apartment|71110.0|                 100|
    |  28134193|Beijing|             Haidian|         Apartment|68984.0|                  60|
    |  22382855|Beijing|            Chaoyang|       Condominium|68980.0|                 100|
    |  35658816|Beijing|            Chaoyang|         Apartment|68088.0|                null|
    |  34071825|Beijing|            Chaoyang|             House|68002.0|                null|
    |  28803519|Beijing|           Dongcheng|       Condominium|65973.0|                  80|
    |  15488817|Beijing|              Zhuang|Serviced apartment|63345.0|                  86|
    |  20748712|Beijing|            Chaoyang|       Condominium|60002.0|                  60|
    |  21942314|Beijing|             Fengtai|             House|60002.0|                 100|
    |  31432793|Beijing|Liang Ma Qiao/San...|         Apartment|60002.0|                 100|
    |  29138170|Beijing|            Chaoyang|       Condominium|60002.0|                null|
    |  29520327|Beijing|            Chaoyang|             House|43798.0|                null|
    |  15526505|Beijing|             Shilipu|         Apartment|39358.0|                 100|
    |  30355531|Beijing|             Fengtai|             House|30001.0|                null|
    |  22035885|Beijing|             Xicheng|    Boutique hotel|26531.0|                 100|
    |  32824344|Beijing|            Chaoyang|Serviced apartment|25998.0|                null|
    |  32825776|Beijing|            Chaoyang|Serviced apartment|25998.0|                null|
    |  33539654|Beijing|             Shilipu|         Apartment|25998.0|                  80|
    |  33943498|Beijing|  Wangfujing/Dongdan|Serviced apartment|25002.0|                null|
    |  33751384|Beijing|             Shilipu|             House|24000.0|                null|
    +----------+-------+--------------------+------------------+-------+--------------------+
    only showing top 20 rows
    
    

<font color='blue'>    
The most expensive listings in beijing is $711,110. It locates in `Chaoyang` district and it is an `Apartment`.
</font>


```python
# calculate the min, max and average
beijing_price.select(f.min('price'), f.max('price'), f.avg('price')).show()
```

    +----------+----------+-----------------+
    |min(price)|max(price)|       avg(price)|
    +----------+----------+-----------------+
    |       0.0|   71110.0|565.5259535417264|
    +----------+----------+-----------------+
    
    

### Compute and visualize the distribution of listing prices in your city.

<font color='blue'>    
A good way to show the distribution of listing prices is the histogram.</font>


```python
beijing_price_df = beijing_price.toPandas()
```


```python
# histogram show the price distribution
plt.figure(figsize = [20,12])
plot = sns.distplot(beijing_price_df.price, kde=False, bins = 4000, hist_kws = {'alpha': 0.9}, color = sns.color_palette()[0])
plt.xticks(np.arange(0, beijing_price_df.price.max(), 100))
plt.xlim(0, 1000)
plt.xlabel('Listing price distribution(frequency)', fontsize=13)
plt.show()
```


![png](output_50_0.png)


The value of a listing is its rating divided by its price.

### Compute and show a dataframe with the 3 highest valued listings in each neighbourhood.

We compute the 3 highest valued listings in each neighbourhood in 4 steps :
1. <font color='blue'> cast the review ratings into integers in order to caculate the `value`. </font>


2. <font color='blue'> sort the `value` in descending way.</font>

3. <font color='blue'>  make a window filter to rank the value in each neigbourhood. </font>

4. <font color='blue'>  slect lisitings with rank number less than 3 in each neighbourhood. </font>




```python
# cast string to integer
beijing_price = beijing_price.withColumn('review_scores_rating', (f.col('review_scores_rating')).cast(IntegerType()))
```


```python
# calculate value
beijing_listing_value = beijing_price.withColumn('value', (f.col('review_scores_rating')/f.col('price')).alias('value'))
```


```python
beijing_listing_value.show()
```

    +----------+-------+--------------------+------------------+-----+--------------------+-------------------+
    |listing_id|   city|       neighbourhood|     property_type|price|review_scores_rating|              value|
    +----------+-------+--------------------+------------------+-----+--------------------+-------------------+
    |   6317579|Beijing|           Dongcheng| Bed and breakfast|220.0|                  99|               0.45|
    |   6622351|Beijing|           Dongcheng|        Guesthouse|555.0|                  95|0.17117117117117117|
    |  12597460|Beijing|             Haidian|         Apartment|525.0|                  97|0.18476190476190477|
    |  13495703|Beijing|            Chaoyang|             House|164.0|                 100| 0.6097560975609756|
    |  13564871|Beijing|Liang Ma Qiao/San...|    Boutique hotel|640.0|                null|               null|
    |  14473988|Beijing|          Shuangjing|         Apartment|525.0|                  94|0.17904761904761904|
    |  14997784|Beijing|           Dongcheng|Serviced apartment|668.0|                  98| 0.1467065868263473|
    |  16430595|Beijing|            Sanlitun|         Apartment|853.0|                 100|0.11723329425556858|
    |  18940394|Beijing|              Xuanwu|          Bungalow|498.0|                  95|0.19076305220883535|
    |  19025271|Beijing|             Haidian|             House|468.0|                  90|0.19230769230769232|
    |  19086556|Beijing|             Haidian|         Apartment|498.0|                  50|0.10040160642570281|
    |  19088267|Beijing|  Jinsong/Panjiayuan|         Apartment|170.0|                null|               null|
    |  19102060|Beijing|            Ahn Jung|         Apartment|121.0|                null|               null|
    |  19506317|Beijing|            Chaoyang|         Apartment|191.0|                 100| 0.5235602094240838|
    |  20929761|Beijing|            Chaoyang|         Apartment|177.0|                  99|  0.559322033898305|
    |  21103240|Beijing|            Chaoyang|         Apartment|910.0|                  93| 0.1021978021978022|
    |  21642531|Beijing|            Chaoyang|Serviced apartment|391.0|                  80|0.20460358056265984|
    |  22005612|Beijing|           Dongcheng|             House|740.0|                  90|0.12162162162162163|
    |  22986438|Beijing|             Beiyuan|         Apartment|199.0|                  50|0.25125628140703515|
    |  23172274|Beijing|            Chaoyang|         Apartment|697.0|                  98| 0.1406025824964132|
    +----------+-------+--------------------+------------------+-----+--------------------+-------------------+
    only showing top 20 rows
    
    


```python
# build a window
sorted_by_value = Window.partitionBy('neighbourhood').orderBy(f.desc('value'))

# rank with window and filter by rank number
beijing_ranked_neigh = beijing_listing_value.withColumn('value_rank', f.rank().over(sorted_by_value))
beijing_ranked_neigh.filter(f.col('value_rank') <= 3).drop('value_rank').orderBy('neighbourhood', f.desc('value')).show()
```

    +----------+-------+-------------------+-------------+-----+--------------------+------------------+
    |listing_id|   city|      neighbourhood|property_type|price|review_scores_rating|             value|
    +----------+-------+-------------------+-------------+-----+--------------------+------------------+
    |  20195162|Beijing|           Ahn Jung|    Apartment| 85.0|                 100|1.1764705882352942|
    |  16695169|Beijing|           Ahn Jung|    Apartment| 85.0|                  95|1.1176470588235294|
    |  17971300|Beijing|           Ahn Jung|        House| 92.0|                  89| 0.967391304347826|
    |  23209150|Beijing|Bei Tai Ping Zhuang|    Apartment| 85.0|                 100|1.1764705882352942|
    |  15697701|Beijing|Bei Tai Ping Zhuang|    Apartment| 85.0|                  99|1.1647058823529413|
    |  16318798|Beijing|Bei Tai Ping Zhuang|    Apartment| 92.0|                 100|1.0869565217391304|
    |  33866823|Beijing|Bei Tai Ping Zhuang|       Hostel| 92.0|                 100|1.0869565217391304|
    |  31556177|Beijing| Beijing University|        House|100.0|                  93|              0.93|
    |  18222384|Beijing| Beijing University|   Guesthouse|135.0|                 100|0.7407407407407407|
    |  35962784|Beijing| Beijing University|        House|142.0|                 100| 0.704225352112676|
    |  34515565|Beijing|            Beiyuan|        House| 71.0|                 100| 1.408450704225352|
    |  38238385|Beijing|            Beiyuan|    Apartment| 92.0|                 100|1.0869565217391304|
    |  25457982|Beijing|            Beiyuan|    Apartment|107.0|                 100|0.9345794392523364|
    |  12190198|Beijing|            Caoqiao|  Condominium| 78.0|                  90|1.1538461538461537|
    |  13922593|Beijing|            Caoqiao|    Apartment|100.0|                 100|               1.0|
    |  33304596|Beijing|            Caoqiao|  Condominium|100.0|                 100|               1.0|
    |  32769153|Beijing|           Chaoyang|  Condominium| 64.0|                 100|            1.5625|
    |  34194662|Beijing|           Chaoyang|        House| 64.0|                 100|            1.5625|
    |  34243820|Beijing|           Chaoyang|        House| 64.0|                 100|            1.5625|
    |  14392809|Beijing|           Chaoyang|    Apartment| 64.0|                 100|            1.5625|
    +----------+-------+-------------------+-------------+-----+--------------------+------------------+
    only showing top 20 rows
    
    

## Trends
Now we want to analyze the "popularity" of your city. The data does not contain the number of bookings per listing, but we have a large number of reviews, and we will assume that this is a good indicator of activity on listings.

### Compute and visualize the popularity(i.e. number of reviews) of your city over time.


```python
# check the data type
reviews.printSchema()
```

    root
     |-- listing_id: string (nullable = true)
     |-- id: string (nullable = true)
     |-- date: string (nullable = true)
     |-- reviewer_id: string (nullable = true)
     |-- reviewer_name: string (nullable = true)
     |-- comments: string (nullable = true)
    
    


```python
# join reviews with beijing's listings data
beijing_popularity = listings_sub.filter(f.col('city') == 'Beijing')\
                                 .join(reviews, ['listing_id'], 'inner')\
                                 .withColumn('date', f.col('date').cast(DateType()))
```

Compute and visualize the popularity (i.e., number of reviews) of your city over time.


```python
# groupby day
beijing_popularity_by_city = beijing_popularity.groupBy('date')\
                                               .agg(f.count('id').alias('review_counts'))\
                                               .orderBy('date')\
                                               .toPandas()
```


```python
beijing_popularity_by_city.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>review_counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010-08-25</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2010-10-13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-06-02</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-06-05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-08-02</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# visulize the number of review_counts by time
dates = np.array(beijing_popularity_by_city['date'], dtype=np.datetime64)
source = ColumnDataSource(data=dict(date=dates, close=beijing_popularity_by_city['review_counts']))

p = figure(plot_height=300, plot_width=800, tools="xpan", toolbar_location=None,
           x_axis_type="datetime", x_axis_location="above",
           background_fill_color="#efefef", x_range=(dates[2000], dates[2284]))

p.line('date', 'close', source=source, legend_label='Beijing popularity over time')
p.yaxis.axis_label = 'Number of reviews'

select = figure(title="Drag the middle and edges of the selection box to change the range above",
                plot_height=130, plot_width=800, y_range=p.y_range,
                x_axis_type="datetime", y_axis_type=None,
                tools="", toolbar_location=None, background_fill_color="#efefef")

range_tool = RangeTool(x_range=p.x_range)
range_tool.overlay.fill_color = "navy"
range_tool.overlay.fill_alpha = 0.2

select.line('date', 'close', source=source)
select.ygrid.grid_line_color = None
select.add_tools(range_tool)
select.toolbar.active_multi = range_tool

output_notebook()
show(column(p, select))
```



<div class="bk-root">
    <a href="https://bokeh.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
    <span id="2817">Loading BokehJS ...</span>
</div>











<div class="bk-root" id="c394da9a-9587-404f-99a7-9bf977bc20af" data-root-id="2818"></div>





### Compute and visualize the popularity of neighbourhoods over time.

If there are many neighbourhoods in your city, you should select a few interesting ones for comparison.

<font color='blue'> We select Chaoyang, Haidian, Fengtai, Dongchen based on the number of listings</font>


```python
beijing_popularity_by_neigh = beijing_popularity.filter((f.col('neighbourhood')=='Chaoyang') | (f.col('neighbourhood')=='Haidian') | (f.col('neighbourhood')=='Fengtai') | (f.col('neighbourhood')=='Dongcheng'))\
                                                .groupBy('neighbourhood', 'date')\
                                                .agg(f.count('id').alias('review_counts'))\
                                                .orderBy('date')\
                                                .toPandas()
```


```python
beijing_popularity_by_neigh.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>neighbourhood</th>
      <th>date</th>
      <th>review_counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Chaoyang</td>
      <td>2010-08-25</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chaoyang</td>
      <td>2010-10-13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dongcheng</td>
      <td>2011-06-02</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Dongcheng</td>
      <td>2011-06-05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Dongcheng</td>
      <td>2011-08-02</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# visualize the popularity of neighbourhoods over time
source1 = beijing_popularity_by_neigh[beijing_popularity_by_neigh['neighbourhood']=='Chaoyang']
source2 = beijing_popularity_by_neigh[beijing_popularity_by_neigh['neighbourhood']=='Haidian']
source3 = beijing_popularity_by_neigh[beijing_popularity_by_neigh['neighbourhood']=='Fengtai']
source4 = beijing_popularity_by_neigh[beijing_popularity_by_neigh['neighbourhood']=='Dongcheng']

p = figure(plot_height=300, plot_width=800, tools="xpan", toolbar_location=None,
           x_axis_type="datetime", x_axis_location="above",
           background_fill_color="#efefef", x_range=(dates[2000], dates[2284]))

p.line(source1['date'], source1['review_counts'], legend_label='Chaoyang popularity over time', line_color='blue')
p.line(source2['date'], source2['review_counts'], legend_label='Haidian popularity over time', line_color='orange')
p.line(source3['date'], source3['review_counts'], legend_label='Fengtai popularity over time', line_color='red')
p.line(source4['date'], source4['review_counts'], legend_label='Dongcheng popularity over time', line_color='green')

p.yaxis.axis_label = 'Number of reviews'

select = figure(title="Drag the middle and edges of the selection box to change the range above",
                plot_height=130, plot_width=800, y_range=p.y_range,
                x_axis_type="datetime", y_axis_type=None,
                tools="", toolbar_location=None, background_fill_color="#efefef")

range_tool = RangeTool(x_range=p.x_range)
range_tool.overlay.fill_color = "navy"
range_tool.overlay.fill_alpha = 0.2

select.line(source1['date'], source1['review_counts'])
select.line(source2['date'], source2['review_counts'])
select.line(source3['date'], source3['review_counts'])
select.line(source4['date'], source4['review_counts'])

select.ygrid.grid_line_color = None
select.add_tools(range_tool)
select.toolbar.active_multi = range_tool

output_notebook()
show(column(p, select))
```



<div class="bk-root">
    <a href="https://bokeh.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
    <span id="5563">Loading BokehJS ...</span>
</div>











<div class="bk-root" id="5cf475f9-cc0f-4bc4-bacf-4f17ed33d880" data-root-id="5564"></div>





### Compute and visualize the popularity of your city by season. 
For example, visualize the popularity of your city per month.


```python
# manipulate data into the format "MM-01-yyyy"
beijing_popularity_by_city_seasonal = beijing_popularity.withColumn('month', f.date_format(f.col('date'), 'MM-01-yyyy'))\
                                                        .groupBy('month')\
                                                        .agg(f.count('id').alias('review_counts'))\
                                                        .orderBy(f.col('month'))\
                                                        .toPandas()
```


```python
beijing_popularity_by_city_seasonal.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>month</th>
      <th>review_counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01-01-2012</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01-01-2013</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01-01-2014</td>
      <td>23</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01-01-2015</td>
      <td>42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01-01-2016</td>
      <td>311</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Turn month column into datetime to fix the formatting issues and make processing easier
beijing_popularity_by_city_seasonal.month = pd.to_datetime(beijing_popularity_by_city_seasonal.month)
```


```python
beijing_popularity_by_city_seasonal['Month_Name'] = beijing_popularity_by_city_seasonal.month.apply(lambda row: row.month_name())
```


```python
beijing_popularity_by_city_seasonal['Year'] = beijing_popularity_by_city_seasonal.month.apply(lambda row: str(row.year))
```


```python
beijing_popularity_by_city_seasonal = beijing_popularity_by_city_seasonal.set_index(['Month_Name', 'Year'])['review_counts'].unstack().loc[:, ['2012', '2013', '2014', '2015', '2016', '2017', '2018']].astype(int).reset_index()
```


```python
beijing_popularity_by_city_seasonal
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Year</th>
      <th>Month_Name</th>
      <th>2012</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
      <th>2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>April</td>
      <td>6</td>
      <td>23</td>
      <td>33</td>
      <td>112</td>
      <td>491</td>
      <td>2002</td>
      <td>4897</td>
    </tr>
    <tr>
      <th>1</th>
      <td>August</td>
      <td>15</td>
      <td>24</td>
      <td>47</td>
      <td>248</td>
      <td>1268</td>
      <td>3782</td>
      <td>8603</td>
    </tr>
    <tr>
      <th>2</th>
      <td>December</td>
      <td>10</td>
      <td>23</td>
      <td>70</td>
      <td>258</td>
      <td>1338</td>
      <td>3000</td>
      <td>7774</td>
    </tr>
    <tr>
      <th>3</th>
      <td>February</td>
      <td>4</td>
      <td>7</td>
      <td>22</td>
      <td>38</td>
      <td>248</td>
      <td>1180</td>
      <td>2874</td>
    </tr>
    <tr>
      <th>4</th>
      <td>January</td>
      <td>3</td>
      <td>7</td>
      <td>23</td>
      <td>42</td>
      <td>311</td>
      <td>1311</td>
      <td>3625</td>
    </tr>
    <tr>
      <th>5</th>
      <td>July</td>
      <td>11</td>
      <td>22</td>
      <td>46</td>
      <td>208</td>
      <td>900</td>
      <td>3090</td>
      <td>7126</td>
    </tr>
    <tr>
      <th>6</th>
      <td>June</td>
      <td>15</td>
      <td>29</td>
      <td>36</td>
      <td>181</td>
      <td>682</td>
      <td>2397</td>
      <td>5819</td>
    </tr>
    <tr>
      <th>7</th>
      <td>March</td>
      <td>11</td>
      <td>25</td>
      <td>26</td>
      <td>64</td>
      <td>324</td>
      <td>1452</td>
      <td>1495</td>
    </tr>
    <tr>
      <th>8</th>
      <td>May</td>
      <td>12</td>
      <td>26</td>
      <td>36</td>
      <td>118</td>
      <td>658</td>
      <td>2391</td>
      <td>5512</td>
    </tr>
    <tr>
      <th>9</th>
      <td>November</td>
      <td>12</td>
      <td>29</td>
      <td>47</td>
      <td>218</td>
      <td>892</td>
      <td>2521</td>
      <td>6616</td>
    </tr>
    <tr>
      <th>10</th>
      <td>October</td>
      <td>12</td>
      <td>36</td>
      <td>62</td>
      <td>314</td>
      <td>1281</td>
      <td>2618</td>
      <td>8419</td>
    </tr>
    <tr>
      <th>11</th>
      <td>September</td>
      <td>9</td>
      <td>32</td>
      <td>51</td>
      <td>251</td>
      <td>1169</td>
      <td>3268</td>
      <td>7321</td>
    </tr>
  </tbody>
</table>
</div>




```python
#visualize the popularity of your city by season
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018']
colors = sns.color_palette().as_hex()[2:len(years)+2]

p = figure(x_range = months, plot_height=250, plot_width = 1000, title="Fruit Counts by Year",
           toolbar_location=None, tools="")

p.vbar_stack(years, x='Month_Name', width=0.9, color=colors, source=beijing_popularity_by_city_seasonal, legend_label=years)

p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.axis.minor_tick_line_color = None
p.outline_line_color = None
p.legend.location = "top_left"
p.legend.orientation = "horizontal"

output_notebook()
show(column(p))
```



<div class="bk-root">
    <a href="https://bokeh.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
    <span id="27453">Loading BokehJS ...</span>
</div>











<div class="bk-root" id="a992b369-bf23-4cc6-8c86-304bed72e669" data-root-id="27454"></div>





<font color='blue'> As the Airbnd is not so popular in Beijing before 2016, it is popularity is not shown obviously in above figure.  </font>

## Reviews
In this part you should determine which words used in reviews that are the most positive. 

The individual reviews do not have a rating of the listing, so we will assume that each review gave the average rating to the listing, i.e., the one on the listing.

You should assign a positivity weight to each word seen in reviews and list the words with the highest weight. It is up to you to decide what the weight should be. For example, it can be a function of the rating on the listing on which it occurs, the number of reviews it occurs in, and the number of unique listings for which it was used to review.

Depending on your choice of weight function, you may also want to do some filtering of words. For example, remove words that only occur in a few reviews.

<font color='blue'> We solve this question in steps:</font>


1. <font color='blue'>join the reviews into the listings data based on the listing's id.</font>
2. <font color='blue'>split the review comments into words and `explode` it into rows. Assume the word score euqals to the review score.</font>
3. <font color='blue'>groupy by distinct word and calculate the mean of word score as well as the word frequecy.</font>
4. <font color='blue'>calculate the weighted score of each word based on the formula below.</font>

$$
weighted\_score = average\_score * \sqrt{word\_frequency}
$$
5. <font color='blue'> show 5 words with the highest weighted score.</font>
6. <font color='blue'> show 5 words with the highest average score but drop the word whose frequency are less than 200 million.</font>
7. <font color='blue'> compare the words list.</font>


```python
# data clean
rated_reviews = listings_sub.join(reviews, ['listing_id'], 'inner')\
                            .withColumn('date', f.col('date').cast(DateType()))\
                            .withColumn('review_scores_rating', (f.col('review_scores_rating')).cast(IntegerType()))\
                            .filter(f.col('review_scores_rating').isNotNull())\
                            .orderBy(f.desc('review_scores_rating'))
```


```python
# calculate weighted score, average score and frequency
weighted_rated_reviews = rated_reviews.withColumn('words', explode(split(f.col('comments'), '\W+').alias('words')))\
                                      .select('words', 'review_scores_rating')\
                                      .groupBy('words')\
                                      .agg(f.mean('review_scores_rating').alias('avg_scores'), f.count('review_scores_rating').alias('frequency'))\
                                      .withColumn('weighted_scores', f.col('avg_scores')*f.col('frequency')**(1/2))\
                                      .orderBy(f.desc('weighted_scores'))\
```


```python
weighted_rated_reviews.show(100)
```

    +-----------+-----------------+---------+------------------+
    |      words|       avg_scores|frequency|   weighted_scores|
    +-----------+-----------------+---------+------------------+
    |        and|95.38446260851613| 56651028| 717929.0631763524|
    |        the|94.78737839063318| 45500093| 639376.4765151071|
    |          a| 94.9671802056149| 37460320| 581245.0728823153|
    |         to| 95.1163928305043| 35362264| 565620.8406016177|
    |           |94.81618909027294| 29747527| 517139.7600633213|
    |        was|94.98309220656003| 26994652|493497.74286926084|
    |         is|95.02753095496094| 24109625| 466599.9340859086|
    |         in|94.91229512144986| 22275819|  447960.082154477|
    |        The|94.94596874803665| 15598380|374987.05716002936|
    |       very| 95.1088069819577| 15197113|370767.17711119284|
    |         of|95.12319003040311| 14630080| 363839.4250530417|
    |        for|94.99795637736182| 14382303| 360270.3153953516|
    |          s| 94.9018364609585| 13925262| 354141.0773476358|
    |          I|95.12058574757597| 12203482| 332289.3318177492|
    |      place|95.24744166661456| 11996775| 329902.4769323235|
    |       with|95.29559016574098| 11143412|318113.35198231135|
    |       stay| 95.4442867301635| 11069948| 317557.7580453187|
    |      great| 95.4029896860036| 10444642|308325.01688483945|
    |         we|95.09696650190183| 10011643|300898.02769293496|
    |        you|95.09179293254158|  9094698|286772.28525679064|
    |  apartment|94.82646076013864|  8509816| 276623.8531842354|
    |         it|94.52071476886938|  8400142| 273949.3731020657|
    |   location|94.77713941782233|  8275721|272650.63721528917|
    |         We|95.53479996573648|  7623269| 263774.1432055377|
    |      clean|95.30230785323009|  6873834| 249863.5681926265|
    |         us|95.35846774238718|  6248897|238375.13234979034|
    |        our|95.47740065772818|  6185534| 237459.3016153283|
    |         de|93.74897452229801|  6350455| 236248.4388482613|
    |         at|95.18798446582971|  6141300|   235891.49758361|
    |        had|95.07513312131093|  5982701| 232549.5970911644|
    |       were|95.09261953650125|  5946510|231887.79340753102|
    |       host| 95.1651470095354|  5710718|227417.18541364872|
    |         as|94.85934620833214|  5478473| 222029.0955895442|
    |       nice|94.80211029246783|  5352623|219331.66364097325|
    |       from|95.05265281867527|  5186408|216469.93229633317|
    |         on|94.78545617625683|  5083945| 213718.5044028016|
    |         It|95.10986162385046|  4923175| 211031.9376673511|
    |        but|93.90564827589859|  4909746|208075.63451729692|
    |       this|94.93024145947574|  4778483|207515.05422092127|
    |  recommend|95.64356330749465|  4672322|206738.86633391897|
    |       that|94.44120294469836|  4782697|206537.03786968024|
    |         so|95.17539793221641|  4542545| 202849.6769625939|
    |        are|95.07367498491193|  4488647|201427.14993116292|
    |        all|95.13279709109176|  4427228|200168.71919574295|
    |      would| 95.2274827012498|  4200737|195175.38742457228|
    |          t|94.24757158348358|  4192238|192971.48278786556|
    |comfortable|95.72393772561357|  3966960|  190655.556193693|
    | everything|95.56136087136093|  3832809| 187085.8384615892|
    |      there| 94.5004643538194|  3896813|186547.19587704944|
    |      again|95.76117241289091|  3745677|185333.79413434947|
    |      Great|94.96335454616879|  3742074|183701.30150652467|
    |         an|95.14065317803181|  3720762|183519.43953982703|
    |     really|95.07656651409276|  3717761| 183321.8466525952|
    |       well|95.35040109951755|  3636504| 181829.5898453963|
    |         la|93.43163484651143|  3777480|181591.29882558843|
    |       good|94.08479450231043|  3657973| 179944.9605218761|
    |         et|94.12369670495197|  3646047|179725.66826057073|
    |    perfect| 95.7042850242088|  3475803| 178426.3371079245|
    |       room|94.34828743789681|  3540076|177517.14901347584|
    |          y|93.98835911243717|  3555399|177222.24776193604|
    |       have|95.01385633593384|  3423055|175789.88572218787|
    |          e|94.20108522118414|  3448145|174923.70449616283|
    |      house|95.71923325060698|  3338718|174899.70636633915|
    |      close| 94.9004603649237|  3396436| 174896.0644278762|
    |         my|95.19648545993361|  3203321|170380.99317156064|
    |       time| 95.2376125554801|  3173435| 169657.5938950928|
    |         be|94.70024204006448|  3198644|169369.04647434072|
    |        not|93.43461413772452|  3232656|167991.59070206375|
    |       home|96.47683896861794|  3004551|167229.48627856103|
    |       easy|95.17947182275599|  3050529|166238.21595214322|
    | definitely|95.83080206157007|  2772256|159559.17754859792|
    |        est| 93.8650850734403|  2877984|159238.56883673763|
    |       walk|95.03518309200224|  2800095|159026.97532524943|
    |        out| 94.8822404317823|  2744346|157182.56212669692|
    |       here|95.69576425772148|  2581531|153755.75759276975|
    |    helpful|95.29366891599118|  2600929|153683.87297580077|
    |         en|93.90334225927072|  2647341|152786.85559092654|
    |       This|  95.239948063156|  2563113|152476.53936398905|
    |       just|95.02492837790035|  2572530| 152411.5132089172|
    |         un| 93.6565633733043|  2632244|151950.20487581272|
    |       also|95.19813965220956|  2537160|151636.02437145615|
    |         tr|94.13388111059297|  2581828|151254.95756859318|
    |      which|94.70748894043716|  2547343|151156.92042574752|
    |restaurants|95.41023770225424|  2498588| 150814.2236322502|
    |         if|94.69323012018941|  2483663|149233.13585447246|
    |       area|95.23842834502614|  2349167|145971.86797675598|
    |       city|95.02954644584405|  2340823|145392.81477611206|
    |      super| 95.6033348691873|  2307077|145212.52814732556|
    |       Very|95.02841004155232|  2319004|144711.88873668524|
    |        has|95.29689232907549|  2303751|144642.69556136854|
    |          n| 93.5364786319922|  2389577| 144591.0896892946|
    |      space| 95.6577555584908|  2256098|143680.94210296762|
    |         by|94.77250781735309|  2289138| 143389.8334229332|
    |     lovely|95.96121838963714|  2225101|143143.16576973384|
    |         me|94.63210411802447|  2263124|142361.53816218747|
    |      quiet|95.62769643965468|  2215375|142333.56248263494|
    |        can|94.93997959458645|  2204317|140956.84128817485|
    |  beautiful|96.39824455807377|  2133252|140795.96412913894|
    |      check|94.41381449813223|  2098491|136769.44978582306|
    |       back| 95.8136207053185|  2036385|136727.91674854665|
    +-----------+-----------------+---------+------------------+
    only showing top 100 rows
    
    

<font color='blue'>In the above table, we can find some words has very weighted score. but they do not show positive or negative emotion in real life. So we will ignore them and concentrate on the `adjactives`. Then, we list the most 5 postive adjactives as below.
</font>
1. great
2. clean
3. nice
4. recommend
5. comfortable


```python
weighted_rated_reviews.orderBy(f.desc('avg_scores')).filter(f.col('frequency') > 2000000).show(100)
```

    +-----------+-----------------+---------+------------------+
    |      words|       avg_scores|frequency|   weighted_scores|
    +-----------+-----------------+---------+------------------+
    |       home|96.47683896861794|  3004551|167229.48627856103|
    |  beautiful|96.39824455807377|  2133252|140795.96412913894|
    |     lovely|95.96121838963714|  2225101|143143.16576973384|
    | definitely|95.83080206157007|  2772256|159559.17754859792|
    |       back| 95.8136207053185|  2036385|136727.91674854665|
    |      again|95.76117241289091|  3745677|185333.79413434947|
    |comfortable|95.72393772561357|  3966960|  190655.556193693|
    |      house|95.71923325060698|  3338718|174899.70636633915|
    |    perfect| 95.7042850242088|  3475803| 178426.3371079245|
    |       here|95.69576425772148|  2581531|153755.75759276975|
    |      space| 95.6577555584908|  2256098|143680.94210296762|
    |  recommend|95.64356330749465|  4672322|206738.86633391897|
    |      quiet|95.62769643965468|  2215375|142333.56248263494|
    |      super| 95.6033348691873|  2307077|145212.52814732556|
    | everything|95.56136087136093|  3832809| 187085.8384615892|
    |         We|95.53479996573648|  7623269| 263774.1432055377|
    |        our|95.47740065772818|  6185534| 237459.3016153283|
    |       stay| 95.4442867301635| 11069948| 317557.7580453187|
    |restaurants|95.41023770225424|  2498588| 150814.2236322502|
    |      great| 95.4029896860036| 10444642|308325.01688483945|
    |        and|95.38446260851613| 56651028| 717929.0631763524|
    |         us|95.35846774238718|  6248897|238375.13234979034|
    |       well|95.35040109951755|  3636504| 181829.5898453963|
    |      clean|95.30230785323009|  6873834| 249863.5681926265|
    |        has|95.29689232907549|  2303751|144642.69556136854|
    |       with|95.29559016574098| 11143412|318113.35198231135|
    |    helpful|95.29366891599118|  2600929|153683.87297580077|
    |      place|95.24744166661456| 11996775| 329902.4769323235|
    |       This|  95.239948063156|  2563113|152476.53936398905|
    |       area|95.23842834502614|  2349167|145971.86797675598|
    |       time| 95.2376125554801|  3173435| 169657.5938950928|
    |      would| 95.2274827012498|  4200737|195175.38742457228|
    |       also|95.19813965220956|  2537160|151636.02437145615|
    |         my|95.19648545993361|  3203321|170380.99317156064|
    |         at|95.18798446582971|  6141300|   235891.49758361|
    |       easy|95.17947182275599|  3050529|166238.21595214322|
    |         so|95.17539793221641|  4542545| 202849.6769625939|
    |       host| 95.1651470095354|  5710718|227417.18541364872|
    |         an|95.14065317803181|  3720762|183519.43953982703|
    |        all|95.13279709109176|  4427228|200168.71919574295|
    |         of|95.12319003040311| 14630080| 363839.4250530417|
    |          I|95.12058574757597| 12203482| 332289.3318177492|
    |         to| 95.1163928305043| 35362264| 565620.8406016177|
    |         It|95.10986162385046|  4923175| 211031.9376673511|
    |       very| 95.1088069819577| 15197113|370767.17711119284|
    |         we|95.09696650190183| 10011643|300898.02769293496|
    |       were|95.09261953650125|  5946510|231887.79340753102|
    |        you|95.09179293254158|  9094698|286772.28525679064|
    |     really|95.07656651409276|  3717761| 183321.8466525952|
    |        had|95.07513312131093|  5982701| 232549.5970911644|
    |        are|95.07367498491193|  4488647|201427.14993116292|
    |       from|95.05265281867527|  5186408|216469.93229633317|
    |       walk|95.03518309200224|  2800095|159026.97532524943|
    |       city|95.02954644584405|  2340823|145392.81477611206|
    |       Very|95.02841004155232|  2319004|144711.88873668524|
    |         is|95.02753095496094| 24109625| 466599.9340859086|
    |       just|95.02492837790035|  2572530| 152411.5132089172|
    |       have|95.01385633593384|  3423055|175789.88572218787|
    |        for|94.99795637736182| 14382303| 360270.3153953516|
    |        was|94.98309220656003| 26994652|493497.74286926084|
    |          a| 94.9671802056149| 37460320| 581245.0728823153|
    |      Great|94.96335454616879|  3742074|183701.30150652467|
    |        The|94.94596874803665| 15598380|374987.05716002936|
    |        can|94.93997959458645|  2204317|140956.84128817485|
    |       this|94.93024145947574|  4778483|207515.05422092127|
    |         in|94.91229512144986| 22275819|  447960.082154477|
    |          s| 94.9018364609585| 13925262| 354141.0773476358|
    |      close| 94.9004603649237|  3396436| 174896.0644278762|
    |        out| 94.8822404317823|  2744346|157182.56212669692|
    |         as|94.85934620833214|  5478473| 222029.0955895442|
    |  apartment|94.82646076013864|  8509816| 276623.8531842354|
    |           |94.81618909027294| 29747527| 517139.7600633213|
    |       nice|94.80211029246783|  5352623|219331.66364097325|
    |        the|94.78737839063318| 45500093| 639376.4765151071|
    |         on|94.78545617625683|  5083945| 213718.5044028016|
    |   location|94.77713941782233|  8275721|272650.63721528917|
    |         by|94.77250781735309|  2289138| 143389.8334229332|
    |      which|94.70748894043716|  2547343|151156.92042574752|
    |        get|94.70643457843138|  2081737|136644.58236776834|
    |         be|94.70024204006448|  3198644|169369.04647434072|
    |         if|94.69323012018941|  2483663|149233.13585447246|
    |         me|94.63210411802447|  2263124|142361.53816218747|
    |         it|94.52071476886938|  8400142| 273949.3731020657|
    |      there| 94.5004643538194|  3896813|186547.19587704944|
    |       that|94.44120294469836|  4782697|206537.03786968024|
    |      check|94.41381449813223|  2098491|136769.44978582306|
    |       room|94.34828743789681|  3540076|177517.14901347584|
    |          t|94.24757158348358|  4192238|192971.48278786556|
    |          e|94.20108522118414|  3448145|174923.70449616283|
    |          d| 94.1569644563994|  2026497| 134037.2260400446|
    |         tr|94.13388111059297|  2581828|151254.95756859318|
    |         et|94.12369670495197|  3646047|179725.66826057073|
    |       good|94.08479450231043|  3657973| 179944.9605218761|
    |        muy|94.05540382498624|  2035040|134174.57471043413|
    |          y|93.98835911243717|  3555399|177222.24776193604|
    |        but|93.90564827589859|  4909746|208075.63451729692|
    |         en|93.90334225927072|  2647341|152786.85559092654|
    |        est| 93.8650850734403|  2877984|159238.56883673763|
    |         de|93.74897452229801|  6350455| 236248.4388482613|
    |         un| 93.6565633733043|  2632244|151950.20487581272|
    +-----------+-----------------+---------+------------------+
    only showing top 100 rows
    
    

<font color='blue'>Then, we sort the average score and drop the words whose frequencies are less than 2 million, the most positive adjactives could be as below:</font>
1. beautiful
2. lovely
3. comfortable
4. perfect
5. recommend
