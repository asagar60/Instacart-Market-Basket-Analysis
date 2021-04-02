# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 00:33:21 2021

@author: asaga
"""

import pandas as pd
import pickle
from datetime import datetime
from f1optimization_faron import get_best_prediction

def get_recommendations(X = None):


    start_time = datetime.now()

    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    today = int(dt_string.split("/")[0])

    #get data from user end
    user_id = int(X['user_id']) #user_id
    order_hour_of_day = int(dt_string.split(" ")[1].split(":")[0]) #current date
    order_dow = datetime.today().weekday() #current day of week




    ulp = pd.read_pickle("user_last_purchase.pkl")
    if user_id not in ulp['user_id'].values:

        #get top 5 products based on hour of day and day of week
        top= pd.read_pickle('top10_products.pkl')
        top_products = top[(top['order_dow']==order_dow) & (top['order_hour_of_day']==order_hour_of_day)]['product_name'].values.tolist()
        top_products = {i: value for i,value in enumerate(top_products)}
        #paths = get_image_paths(top5_products)
        predictions={}
        predictions['top'] =  top_products

        del ulp, top,now, today, dt_string, order_dow, order_hour_of_day

        end_time = datetime.now()
        difference = end_time - start_time
        #print("Total Time : {} seconds".format(difference))
        time = "{}".format(difference)

        return predictions,time

    user_last_order_date = ulp[ulp['user_id']==user_id]['date'].values.tolist()[0]

    days_since_prior_order = today - int(user_last_order_date.split('-')[-1])
    del ulp, now, today, dt_string, user_last_order_date
    #featurization

    hour_rate = pd.read_pickle("hour_reorder_rate.pkl")
    day_rate = pd.read_pickle("day_reorder_rate.pkl")
    p_days_rate = pd.read_pickle("p_days_since_prior_order_reorder_rate.pkl")
    u_days_rate = pd.read_pickle("u_days_since_prior_order_reorder_rate.pkl")
    up_days_rate = pd.read_pickle("days_since_prior_reorder_rate.pkl")

    merged_up_features = pd.read_pickle("merged_user_product_features.pkl")

    featurized_data = merged_up_features[merged_up_features['user_id']==user_id]

    hour_r = hour_rate[hour_rate['order_hour_of_day']==order_hour_of_day]
    day_r = day_rate[day_rate['order_dow'] == order_dow]
    p_days = p_days_rate[p_days_rate['days_since_prior_order']==days_since_prior_order]
    u_days = u_days_rate[(u_days_rate['user_id']==user_id) & (u_days_rate['days_since_prior_order']==days_since_prior_order)]
    
    if p_days.empty:
        #handle
        p_days = pd.DataFrame(columns = p_days.columns)
        products_x = pd.read_pickle('product_mappings.pkl')
        p_days['product_id'] = products_x['product_id']
        p_days['days_since_prior_order'] = days_since_prior_order
        p_days['p_days_since_prior_order_reorder_rate']=0.0
    
    if u_days.empty:
        #handle
        u_days = pd.DataFrame(columns = u_days.columns)
        df2 = {'user_id': user_id, 'days_since_prior_order': days_since_prior_order, 'u_days_since_prior_order_reorder_rate': 0}
        u_days = u_days.append(df2, ignore_index = True)
        del df2

    up_days = up_days_rate[(up_days_rate['user_id']==user_id) & (up_days_rate['days_since_prior_order']==days_since_prior_order)]

    if up_days.empty:
        #handle
        up_days = pd.DataFrame(columns = up_days_rate.columns)
        products_x = pd.read_pickle('product_mappings.pkl')
        up_days['product_id'] = products_x['product_id']
        up_days['user_id'] = user_id
        up_days['days_since_prior_order'] = days_since_prior_order
        up_days['days_since_prior_reorder_rate']=0
        del products_x


    #print(up_days_rate[up_days_rate['user_id']==user_id])
    #print(u_days_rate[u_days_rate['user_id']==user_id])
    #print(day_rate)
    del merged_up_features, hour_rate, day_rate, p_days_rate, u_days_rate, up_days_rate

    featurized_data = pd.merge(featurized_data, up_days, on = ['user_id', 'product_id'])

    featurized_data = pd.merge(featurized_data, hour_r, on = 'product_id')
    featurized_data = pd.merge(featurized_data, day_r, on = 'product_id')
    featurized_data = pd.merge(featurized_data, p_days, on = ['product_id','days_since_prior_order'])
    featurized_data = pd.merge(featurized_data, u_days, on = ['user_id','days_since_prior_order'])
    featurized_data = featurized_data[['user_id', 'product_id', 'u_p_order_rate', 'u_p_reorder_rate',\
                                       'u_p_avg_position', 'u_p_orders_since_last', 'max_streak', 'user_reorder_rate',\
                                       'user_unique_products', 'user_total_products', 'user_avg_cart_size', \
                                       'user_avg_days_between_orders', 'user_reordered_products_ratio', \
                                       'product_reorder_rate', 'avg_pos_incart', 'p_reduced_feat_1', 'p_reduced_feat_2', \
                                       'p_reduced_feat_3', 'aisle_id', 'department_id', 'aisle_reorder_rate', \
                                       'dept_reorder_rate', 'order_dow', 'order_hour_of_day', 'days_since_prior_order',\
                                       'hour_reorder_rate', 'day_reorder_rate', 'p_days_since_prior_order_reorder_rate',\
                                       'u_days_since_prior_order_reorder_rate', 'days_since_prior_reorder_rate']]

    del up_days,u_days,p_days,day_r,hour_r

    #model
    with open("catboost_v3.pkl", "rb") as f:
        model = pickle.load(f)
    #model = pickle.load(open("catboost_v3.pkl", "rb"))
    data = featurized_data.drop(['user_id', 'product_id'], axis = 1)
    ypred = model.predict_proba(data)
    ypred = ypred[:,-1] #get probabilities of class 1
    del data,model

    #run faron's optimization code to get most probable set of products which might be reordered
    recommended_products = get_best_prediction(featurized_data['product_id'].tolist(), ypred.tolist(), None)
    recommended_products = recommended_products.replace("None", "")
    recommended_products = list(map(int, recommended_products.split()))
    products_x = pd.read_pickle('product_mappings.pkl')
    recommended_products = products_x.loc[products_x['product_id'].isin(recommended_products)]['product_name'].values.tolist()
    recommended_products = {i: value for i,value in enumerate(recommended_products)}

    predictions= {}
    predictions['recommend'] = recommended_products

    end_time = datetime.now()
    difference = end_time - start_time
    #print("Total Time : {} seconds".format(difference))
    time = "{}".format(difference)

    del featurized_data, products_x
    return predictions, time