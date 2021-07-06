import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import plotly as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import skew, norm, probplot, boxcox
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")


# Page layout
## Page expands to full width
st.set_page_config(layout="wide")



img1 = mpimg.imread('instacart-vector-logo.png')
st.image(img1, width = 500)

st.write("""
   # Instacart Market Analysis
   
   You can come across website of Instacart at [Instacart](https://instacart.careers/)

   ### Made by : **ATOMFAIR Group 5**. 
   

   """)

st.markdown("""
### Let's see some deep understand of both concepts:: 

#### 1. **Customer segmentation**: 
    - The problem of uncovering information about a firm's customer base, based on their interactions with the business. 
    - In most cases this interaction is in terms of their purchase behavior and patterns. 
    - We explore some of the ways in which this can be used..
#### 2. **Market basket analysis:**
    - A method to gain insights into granular behavior of customers. 
    - This is helpful in devising strategies which uncovers deeper understanding of purchase decisions taken by the customers. 
    - This is interesting as a lot of times even the customer will be unaware of such biases or trends in their purchasing behavior.
""")

## Get data
cus_orderdetail_df = pd.read_csv('cus_orderdetail_df.csv')
cluster_centers = pd.read_csv('cluster_center.csv')
products_reordered = pd.read_csv('products_reordered.csv')
product_cluster_center = pd.read_csv('product_cluster_center.csv')
# Page layout (continued)
## Divide page to 3 columns (col1 = sidebar, col2 and col3 = page contents)
col1 = st.sidebar
col2, col3 = st.beta_columns((2,1))

img2 = mpimg.imread('DataCracy.png')
col1.image(img2, width = 280)

col1.header('Analysis Activate')
analysis_active = col1.selectbox('Select Analysis activate ', ('Overview Data',\
                                                               'Customer Segmentation', 'Product Segmentation'))

st.set_option('deprecation.showPyplotGlobalUse', False)

if analysis_active == 'Overview Data':
    col1.header('Dataset')
    dataset = col1.selectbox('Select Dataset', ('Order', 'Order Product', \
                                                  'Product', 'Department','Aisles'))


    if dataset == 'Order':

        order = pd.read_csv('orders.csv')
        col2.subheader('Information of The Order Dataset')

        col2.write("""
                - Order_id: the ID of the Order.
                - User_id: the ID of the User.
                - Eval_set: the dataset type of data row.
                - Order_num: the order of the order.
                - Order_dow: order date by the day of the week.
                - Order_hour_of_day: order hour at day.
                - Days_since_prior_order: the days since the prior order.

                """)
        st.markdown('Datasize of **{}** are **{}** rows and **{}** columns'.format('Order',3421083, 7))
        st.markdown('Dateset inclued: **{}** User ID and **{}** Orders'.format(206209, 3421083))
        st.dataframe(order.head(10))

    elif dataset == 'Order Product':

        order_product_df = pd.read_csv('order_products__train.csv')
        col2.subheader('Information of The Order Product Dataset')

        col2.write("""
                - Order_id: the ID of the Order.
                - Product_id: the ID of the Product.
                - Add_to_cart_order: the line order in the orders.
                - Reordered: Is 1 if the product in the order is ordered otherwise 0.

                """)
        st.markdown('Datasize of **{}** are **{}** rows and **{}** columns'.format('Order Product',1384617,4))
        st.markdown('The sample order')
        st.dataframe(order_product_df[order_product_df['order_id']==1])


    elif dataset == 'Product':

        product_df = pd.read_csv('products.csv')
        col2.subheader('Information of The Product Dataset')

        col2.write("""
                - Product_id: the ID Product.
                - Product_name: The name Product.
                - Aisle_id: the Aisle ID.
                - Department: the department ID.

                """)
        st.markdown('Dataset of **{}** are inclued **{}** product'.format(dataset,len(product_df['product_id'].unique())))

        st.dataframe(product_df.head())


    elif dataset == 'Department':

        department_df = pd.read_csv('departments.csv')
        col2.subheader('Information of The Department Dataset')

        col2.write("""
                - Department_id: the ID Department.
                - Department: the department name.

                """)
        st.markdown('Dataset of **{}** are inclued **{}** department'.format(dataset,len(department_df['department_id'].unique())))

        st.dataframe(department_df)

    else:
        aisle = pd.read_csv('aisles.csv')
        col2.subheader('Information of The Aisle Dataset')

        col2.write("""
                        - Aisle_id: the ID Aisle.
                        - Aisle: the Aisle name.

                        """)
        st.markdown('Dataset of **{}** are inclued **{}** aisle'.format(dataset, len(aisle['aisle_id'].unique())))

        st.dataframe(aisle)

elif analysis_active == 'Customer Segmentation':

    # col1.header('The information type')
    # segment_active = col1.selectbox('Select information type', ('Overview','Segmentation'))

    col1.header('The Columns')
    feature_cus_seg = col1.selectbox('Select Column', ( 'Frequency','Recency','Reordered_0', \
                                                'Reordered_1','Total_line'))
    col2.write("""
    ### Customer Segmentation
     - Segment customers based on frequency information, recency, total line of the orders, and the total line by the re-ordered status.
     - The Datafarm after processed, detail as below:

            - User_id    : The ID User.
            - Reordered_0: The total line has re-ordered status is 0 that customer has.
            - Reordered_1: The total line has re-ordered status is 1 that customer has.
            - Total_line : The total line of the orders that customer has.
            - Frequency  : The total orders of the customer.
            - Recency    : The avg of the days since the prior order by each customer.

    """)

    col2.dataframe(cus_orderdetail_df.iloc[:5,0:6])

    #st.dataframe(cluster_centers)

    feature_selected = ['Frequency','Recency','Reordered_0','Reordered_1','Total_line']
    feature          = ['fequency','recency','reordered_0', 'reordered_1','total_line']

    st.markdown("***")

    # if segment_active == 'Overview':
    #
    #     idx = feature_selected.index(feature_cus_seg)
    #     measure = feature[idx]
    #
    #     st.set_option('deprecation.showPyplotGlobalUse', False)
    #     def Overview (feature_cus_seg, measure):
    #
    #         sns.set(style='darkgrid', font_scale=1.0, rc={"figure.figsize": [14, 6]})
    #
    #         f, ax = plt.subplots(1, 2, figsize=(20, 7))
    #
    #         # Get the fitted parameters used by the function
    #         (mu, sigma) = norm.fit(cus_orderdetail_df[measure])
    #
    #         sns.set(style='darkgrid', font_scale=1.0)
    #
    #         # Kernel Density plot
    #         sns.distplot(cus_orderdetail_df[measure], fit=norm, ax=ax[0])
    #         ax[0].set_title(feature_cus_seg + ' Distribution ( mu = {:.2f} and sigma = {:.2f} )'.format(mu, sigma), loc='center')
    #         ax[0].set_xlabel(feature_cus_seg)
    #         ax[0].set_ylabel('Frequency')
    #
    #         # QQ plot
    #         res = probplot(cus_orderdetail_df[measure], plot=ax[1])
    #         ax[1].set_title(
    #             measure + ' Probability Plot (skewness: {:.6f} and kurtosis: {:.6f} )'.format(cus_orderdetail_df[measure].skew(),
    #                                                                                           cus_orderdetail_df[measure].kurt()),
    #             loc='center')
    #
    #         st.pyplot()
    #
    #     ## Run function when been gotten the parameters out
    #     Overview(feature_cus_seg, measure)
    #
    # else:

    ## Sidebar - Cluster selections
    selected_cluster = col1.selectbox('Select information type', ('Clusters_2', 'Clusters_5','Clusters_8'))

    label = list(cluster_centers[cluster_centers['Number_Cluster'] == selected_cluster]['Cluster'])
    k  = len(label)
    column_label = str(selected_cluster).lower()

    list_target = []
    for val in feature_selected:
        if (feature_cus_seg not in list_target) and val != feature_cus_seg:
            idx = feature_selected.index(val)
            feature_column = feature[idx]
            list_target.append(feature_column)

    idx = feature_selected.index(feature_cus_seg)
    feature_column = feature[idx]

    ## Show information about cluster centroid
    col2.markdown('** The Cluster Centroid information**')
    col2.dataframe(cluster_centers[cluster_centers['Number_Cluster'] == selected_cluster].iloc[:, :6])

    col2.markdown("***")

    col2.markdown('** Conclusion about the clusters: **')
    col2.write("""

        ** Cluster 0: **
        - Segmentation of customers with high purchasing frequency and the low time intervals between purchases.
        - Segmentation of customers with high and frequent shopping values. This is an important customer group for the company/business.

        ** Cluster 1: **
        -  Segmentation of customers with a low frequency of purchases and large time intervals between purchases.

            """)
    col2.markdown('** Recommendations about the clusters: **')
    col2.write("""
    - From the specific information of each customer segmentation and based on the company's development strategy, some recommendations are made:

        ** 1. ** To be more effective needed to combining with other factors such as customer information, products, etc...

        ** 2. ** With segmentation of customers with high and frequent shopping value needed to care, after-sales, promotions, and gratitude program for loyal customers.

        ** 3. ** With segmentation of customers has low shopping value and infrequently needed to find out the reason of this issue. Maybe they are promotion hunters and do not buy fixed products in one place...

            """)
    col2.markdown('***')

    st.set_option('deprecation.showPyplotGlobalUse', False)
    ## Pie chart show information of each Cluster
    cmap = plt.get_cmap('Spectral')
    colors = [cmap(i) for i in np.linspace(0, 1, 8)]


    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'domain'}]])
    fig.add_trace(go.Pie(labels=cus_orderdetail_df[column_label].index, values=cus_orderdetail_df[column_label].value_counts(), marker=dict(colors=colors\
                                                                                       , line=dict(color='#FFF',\
                                                                                                   width=2)),\
                         domain={'x': [0.0, .4], 'y': [0.0, 1]}, name="Cluster"\
                         , showlegend=False, textinfo='label+percent'), 1,1)

    fig.update_layout(height=600,
                      width=1000,
                      autosize=False,
                      paper_bgcolor='rgb(233,233,233)',
                      annotations=[dict(text='Cluster', x=0.50, y=0.5, font_size=20, showarrow=False)],
                      title_text='Distribute of each Cluster  in Dataset')

    # fig = go.Figure(data= data, layout=layout)
    fig.update_traces(hole=.4, hoverinfo="label+percent+name+value")
    fig = go.Figure(fig)
    col2.plotly_chart(fig, filename='transparent-background')
##################################################################################################


    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    fig.add_trace(go.Pie(labels=cus_orderdetail_df[column_label].index,
                         values=cus_orderdetail_df.groupby([column_label])[feature[1]].sum() \
                         , marker=dict(colors=colors, line=dict(color='#FFF', width=2)), \
                         domain={'x': [0.0, .4], 'y': [1.0, 1]}, name="Recency" \
                         , showlegend=True, textinfo='label+percent'), 1, 1)

    fig.add_trace(go.Pie(labels=cus_orderdetail_df[column_label].index,
                         values=cus_orderdetail_df.groupby([column_label])[feature[0]].sum()\
                           , marker=dict(colors=colors , line=dict(color='#FFF',  width=2)), \
                         domain={'x': [0.0, .4], 'y': [1.0, 1]} , name="Frequency"\
                         , showlegend=True, textinfo='label+percent'),1,2)

    fig.update_layout(height=600,
                       width=1000,
                       autosize=False,
                       paper_bgcolor='rgb(233,233,233)',
                       annotations=[dict(text='Recency', x=0.18, y=0.5, font_size=20, showarrow=False),
                                    dict(text='Frequency', x=0.83, y=0.5, font_size=20, showarrow=False)],
                       title_text='Distribute of Recency and Frequency by Cluster in Dataset')

    # fig = go.Figure(data= data, layout=layout)
    fig.update_traces( hole=.4, hoverinfo="label+percent+name+value")
    fig = go.Figure(fig)
    col2.plotly_chart(fig, filename='transparent-background')

    ##############################################################################

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    fig.add_trace(go.Pie(labels=cus_orderdetail_df[column_label].index,
                         values=cus_orderdetail_df.groupby([column_label])[feature[2]].sum() \
                         , marker=dict(colors=colors, line=dict(color='#FFF', width=2)), \
                         domain={'x': [0.0, .4], 'y': [1.0, 1]}, name="Reordered_0" \
                         , showlegend=True, textinfo='label+percent'), 1, 1)

    fig.add_trace(go.Pie(labels=cus_orderdetail_df[column_label].index,
                         values=cus_orderdetail_df.groupby([column_label])[feature[3]].sum() \
                         , marker=dict(colors=colors, line=dict(color='#FFF', width=2)), \
                         domain={'x': [0.0, .4], 'y': [1.0, 1]}, name="Reordered_1" \
                         , showlegend=True, textinfo='label+percent'), 1, 2)

    fig.update_layout(height=600,
                      width=1000,
                      autosize=False,
                      paper_bgcolor='rgb(233,233,233)',
                      annotations=[dict(text='Reordered_0', x=0.16, y=0.5, font_size=20, showarrow=False),
                                   dict(text='Reordered_1', x=0.85, y=0.5, font_size=20, showarrow=False)],
                      title_text='Distribute of Non-Reordered and Re-Ordered by Cluster in Dataset')

    # fig = go.Figure(data= data, layout=layout)
    fig.update_traces(hole=.4, hoverinfo="label+percent+name+value")
    fig = go.Figure(fig)
    col2.plotly_chart(fig, filename='transparent-background')

    if col1.checkbox("Clustering Detail"):

        col2.markdown('** The detailed information of each Cluster **')
        col2.markdown('***')
        def cluster_visual(column_label,k, feature_column, target):

            sns.set(style='darkgrid', font_scale=1.0, rc={"figure.figsize": [14, 6]})

            for i in range(k):
                plt.scatter(cus_orderdetail_df[cus_orderdetail_df[column_label] == i][feature_column], \
                            cus_orderdetail_df[cus_orderdetail_df[column_label] == i][target], s=20,
                            label='Cluster ' + str(i + 1))

                plt.scatter(cluster_centers[(cluster_centers['Number_Cluster'] == selected_cluster) & (
                        cluster_centers['Cluster'] == i)][feature_column], \
                            cluster_centers[(cluster_centers['Number_Cluster'] == selected_cluster) & (
                                    cluster_centers['Cluster'] == i)][target], s=300 \
                            , c='black')
            plt.xlabel(feature_column)
            plt.ylabel(target)
            plt.title('Cluster of ' + feature_cus_seg + ' with ' + target)
            plt.legend()
            col2.pyplot()


        ## Visualization between the features each other
        for idx,val in enumerate(list_target):
            cluster_visual(column_label, k, feature_column, val)

    if col1.checkbox("Features after clustering"):

        col2.markdown('** The information of Features after Clustering **')
        col2.markdown('***')
        ## Box plot to show the different of the features in each cluster
        n_clusters = k


        x_data = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7']
        colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)',
                  'rgba(22, 80, 57, 0.5)', 'rgba(127, 65, 14, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)']

        #cutoff_quantile = 95
        cl = 'clusters_' + str(n_clusters)
        for fild in range(0, len(feature)):
            field_to_plot = feature[fild]
            y_data = list()
            ymax = 0
            for i in np.arange(0, n_clusters):
                y0 = cus_orderdetail_df[cus_orderdetail_df[cl] == i][field_to_plot].values
                # y0 = y0[y0 < np.percentile(y0, cutoff_quantile)]
                # if ymax < max(y0): ymax = max(y0)
                y_data.insert(i, y0)

            traces = []

            for xd, yd, cls in zip(x_data[:n_clusters], y_data, colors[:n_clusters]):
                traces.append(go.Box(y=yd, name=xd, boxpoints=False, jitter=0.5, whiskerwidth=0.2, fillcolor=cls,
                                     marker=dict(size=1, ),
                                     line=dict(width=1),
                                     ))

            layout = go.Layout(
                yaxis=dict(autorange=True, showgrid=True, zeroline=True,
                           dtick=int(ymax / 10),
                           gridcolor='white', gridwidth=0.1, zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2, ),
                margin=dict(l=40, r=30, b=50, t=50, ),
                paper_bgcolor='white',
                plot_bgcolor='rgba(93, 164, 214, 0.5)',
                showlegend=False
            )

            fig = go.Figure(data=traces, layout=layout)
            fig.update_layout(
                title= 'Difference in ' + field_to_plot + ' with ' + str(n_clusters) + ' Clusters.'
            )
            col2.plotly_chart(fig)

elif analysis_active == 'Product Segmentation':

    # col1.header('The information type')
    # segment_active_product = col1.selectbox('Select information type', ('Overview','Segmentation'))

    col1.header('The Columns')
    feature_product_seg = col1.selectbox('Select Column', ('Frequency', 'Reordered_0', 'Reordered_1', \
                                                       'Ordered_first', 'Ordered_last'))
    col2.write("""
        ### Product Segmentation
         - Segment Product based on frequency information, recency -- first and last, the total line by the re-ordered status.
         - The Datafarm after processed, detail as below:

                - Product_id    : The ID Product.
                - Frequency     : The total number of order lines of the product.
                - Reordered_0   : The total line has re-ordered status is 0 that product has.
                - Reordered_1   : The total line has re-ordered status is 1 that product has.
                - Ordered_first : The number of product purchases in the customer's first order.
                - Ordered_last  : The number of product purchases in the customer's last order.
                - Department    : The Department name.
                - Aisle         : The Aisle name.

        """)
    col2.dataframe(products_reordered.iloc[:5, [0,1,2,3,4,5,10,11]])
    col2.markdown("***")

    # if segment_active_product == 'Overview':
    #     feature_product = str(feature_product_seg).lower()
    #     if (feature_product_seg == 'Department') | (feature_product_seg == 'Aisle'):
    #
    #         def make_count(df, col):
    #             fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    #             g = sns.countplot(x=col, order = df[col].value_counts().head(21).index,data=df)
    #             g.set_ylabel("Number")
    #             g.set_title('Distributed of ' + col)
    #             for label in ax.get_xticklabels():
    #                 label.set_rotation(90)
    #             plot_dict = {}
    #             val_counts = dict(df[col].value_counts().head(21).sort_index())
    #             for k, v in val_counts.items():
    #                 if k in val_counts:
    #                     plot_dict[val_counts[k]] = val_counts[k]
    #                 else:
    #                     plot_dict[0] = 0
    #             for x in g.patches:
    #                 height = x.get_height()
    #                 g.text(x.get_x() + x.get_width() / 2.0, height, plot_dict[height] \
    #                        , ha="center", va="bottom", fontsize=8, weight="semibold", size="small")
    #             col2.pyplot()
    #         make_count(products_reordered, feature_product)
    #     else:
    #
    #         def Overview(feature_cus_seg, measure):
    #
    #             sns.set(style='darkgrid', font_scale=1.0, rc={"figure.figsize": [14, 6]})
    #
    #             f, ax = plt.subplots(1, 2, figsize=(20, 7))
    #
    #             # Get the fitted parameters used by the function
    #             (mu, sigma) = norm.fit(products_reordered[measure])
    #
    #             sns.set(style='darkgrid', font_scale=1.0)
    #
    #             # Kernel Density plot
    #             sns.distplot(products_reordered[measure], fit=norm, ax=ax[0])
    #             ax[0].set_title(feature_cus_seg + ' Distribution ( mu = {:.2f} and sigma = {:.2f} )'.format(mu, sigma),
    #                             loc='center')
    #             ax[0].set_xlabel(feature_cus_seg)
    #             ax[0].set_ylabel('Frequency')
    #
    #             # QQ plot
    #             res = probplot(products_reordered[measure], plot=ax[1])
    #             ax[1].set_title(
    #                 measure + ' Probability Plot (skewness: {:.6f} and kurtosis: {:.6f} )'.format(
    #                     products_reordered[measure].skew(),
    #                     products_reordered[measure].kurt()),
    #                 loc='center')
    #
    #             st.pyplot()
    #
    #         Overview(feature_product_seg, feature_product)
    # else:

    product_column = ['frequency', 'reordered_0', 'reordered_1', 'ordered_first', 'ordered_last']
    feature_product = str(feature_product_seg).lower()

    ## Sidebar - Cluster selections
    selected_cluster = col1.selectbox('Select information type', ('Clusters_5','Clusters_8'))


    label = list(product_cluster_center[product_cluster_center['Number_Cluster'] == selected_cluster]['Cluster'])
    k  = len(label)
    column_label = str(selected_cluster).lower()


    list_target = []
    for val in product_column:
        if (product_column not in list_target) and val != feature_product:
            list_target.append(val)

    ## Show information about cluster centroid
    col2.markdown('** The Cluster Centroid information**')
    col2.dataframe(product_cluster_center[product_cluster_center['Number_Cluster'] == selected_cluster].iloc[:, :6])
    col2.markdown("***")

    col2.markdown('** Conclusion about the clusters: **')
    col2.write("""

        ** Cluster 2: **
        - A product segmentation that only accounts for nearly ** 11% ** of the number of products, but accounts for ** 94% ** 
        of the total number of product lines in the order.

        ** Cluster 0: **
        -  Is the product segmentation that accounts for ** 44% ** of the product volume, but only ** 5% ** of the total number of lines in the orders.

        ** Cluster 1 **, ** 3 **, and ** 4 :** 

        - Accounting for about ** 45% ** but only accounting for nearly ** 1% ** 
        of the total number of lines in the orders.

                    """)
    col2.markdown('** Recommendations about the clusters: **')
    col2.write("""
    - From the specific information of each product segmentation and based on the company's development strategy, some recommendations are made:

        ** 1.  ** With ** Cluster 2 ** what needs to be done is to maintain the quality of the inputs and change the models to match the market needs, 
        new marketing methods to attract and retain customers in this product group.

        ** 2. ** With the remaining segments, it is necessary to consider eliminating products that are not really 
        effective to invest in product segments that bring higher profits.
                    """)
    col2.markdown('***')

    st.set_option('deprecation.showPyplotGlobalUse', False)
    ## Pie chart show information of each Cluster
    cmap = plt.get_cmap('Spectral')
    colors = [cmap(i) for i in np.linspace(0, 1, 8)]

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    fig.add_trace(go.Pie(labels=products_reordered[column_label].index, values=products_reordered[column_label].value_counts(), marker=dict(colors=colors\
                                                                                       , line=dict(color='#FFF',\
                                                                                                   width=2)),\
                         domain={'x': [0.0, .4], 'y': [0.0, 1]}, name="Cluster"\
                         , showlegend=False, textinfo='label+percent'), 1,1)
    fig.add_trace(go.Pie(labels=products_reordered[column_label].index,
                         values=products_reordered.groupby([column_label])[product_column[0]].sum() \
                         , marker=dict(colors=colors, line=dict(color='#FFF', width=2)), \
                         domain={'x': [0.0, .4], 'y': [1.0, 1]}, name="Frequency" \
                         , showlegend=True, textinfo='label+percent'), 1, 2)

    fig.update_layout(height=600,
                      width=1000,
                      autosize=False,
                      paper_bgcolor='rgb(233,233,233)',
                      annotations=[dict(text='Cluster', x=0.18, y=0.5, font_size=20, showarrow=False),
                                   dict(text='Frequency', x=0.83, y=0.5, font_size=20, showarrow=False)],
                      title_text='Distribute of Cluster and Frequency in Dataset')

    # fig = go.Figure(data= data, layout=layout)
    fig.update_traces(hole=.4, hoverinfo="label+percent+name+value")
    fig = go.Figure(fig)
    col2.plotly_chart(fig, filename='transparent-background')
##################################################################################################


    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    fig.add_trace(go.Pie(labels=products_reordered[column_label].index,
                         values=products_reordered.groupby([column_label])[product_column[3]].sum() \
                         , marker=dict(colors=colors, line=dict(color='#FFF', width=2)), \
                         domain={'x': [0.0, .4], 'y': [1.0, 1]}, name="Ordered_first" \
                         , showlegend=True, textinfo='label+percent'), 1, 1)

    fig.add_trace(go.Pie(labels=products_reordered[column_label].index,
                         values=products_reordered.groupby([column_label])[product_column[4]].sum()\
                           , marker=dict(colors=colors , line=dict(color='#FFF',  width=2)), \
                         domain={'x': [0.0, .4], 'y': [1.0, 1]} , name="Ordered_last"\
                         , showlegend=True, textinfo='label+percent'),1,2)

    fig.update_layout(height=600,
                       width=1000,
                       autosize=False,
                       paper_bgcolor='rgb(233,233,233)',
                       annotations=[dict(text='Ordered_first', x=0.15, y=0.5, font_size=20, showarrow=False),
                                    dict(text='Ordered_last', x=0.85, y=0.5, font_size=20, showarrow=False)],
                       title_text='Distribute of Ordered first and Ordered last by Cluster in Dataset')

    # fig = go.Figure(data= data, layout=layout)
    fig.update_traces( hole=.4, hoverinfo="label+percent+name+value")
    fig = go.Figure(fig)
    col2.plotly_chart(fig, filename='transparent-background')

    ##############################################################################

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    fig.add_trace(go.Pie(labels=products_reordered[column_label].index,
                         values=products_reordered.groupby([column_label])[product_column[1]].sum() \
                         , marker=dict(colors=colors, line=dict(color='#FFF', width=2)), \
                         domain={'x': [0.0, .4], 'y': [1.0, 1]}, name="Reordered_0" \
                         , showlegend=True, textinfo='label+percent'), 1, 1)

    fig.add_trace(go.Pie(labels=products_reordered[column_label].index,
                         values=products_reordered.groupby([column_label])[product_column[2]].sum() \
                         , marker=dict(colors=colors, line=dict(color='#FFF', width=2)), \
                         domain={'x': [0.0, .4], 'y': [1.0, 1]}, name="Reordered_1" \
                         , showlegend=True, textinfo='label+percent'), 1, 2)

    fig.update_layout(height=600,
                      width=1000,
                      autosize=False,
                      paper_bgcolor='rgb(233,233,233)',
                      annotations=[dict(text='Reordered_0', x=0.16, y=0.5, font_size=20, showarrow=False),
                                   dict(text='Reordered_1', x=0.85, y=0.5, font_size=20, showarrow=False)],
                      title_text='Distribute of Non-Reordered and Re-Ordered by Cluster in Dataset')

    # fig = go.Figure(data= data, layout=layout)
    fig.update_traces(hole=.4, hoverinfo="label+percent+name+value")
    fig = go.Figure(fig)
    col2.plotly_chart(fig, filename='transparent-background')


    if col1.checkbox("Clustering Detail"):

        col2.markdown('** The detailed information of each Cluster **')
        col2.markdown('***')

        def cluster_visual(column_label,k, feature_column, target):

            sns.set(style='darkgrid', font_scale=1.0, rc={"figure.figsize": [14, 6]})

            for i in range(k):
                cutoff_data_feature = products_reordered[products_reordered[column_label] == i][feature_column].quantile(0.70)
                cutoff_data_target = products_reordered[products_reordered[column_label] == i][target].quantile(0.70)
                if cutoff_data_feature > cutoff_data_target :
                    plt.scatter(products_reordered[(products_reordered[column_label] == i) & (products_reordered[feature_column] < cutoff_data_feature)][feature_column], \
                                products_reordered[(products_reordered[column_label] == i) & (products_reordered[feature_column] < cutoff_data_feature)][target], s=20,label='Cluster ' + str(i + 1))
                else:
                    plt.scatter(products_reordered[(products_reordered[column_label] == i) & (
                                products_reordered[target] < cutoff_data_target)][feature_column], \
                                products_reordered[(products_reordered[column_label] == i) & (
                                            products_reordered[target] < cutoff_data_target)][target], s=20,label='Cluster ' + str(i + 1))

                plt.scatter(product_cluster_center[(product_cluster_center['Number_Cluster'] == selected_cluster) & (
                        product_cluster_center['Cluster'] == i)][feature_column], \
                            product_cluster_center[(product_cluster_center['Number_Cluster'] == selected_cluster) & (
                                    product_cluster_center['Cluster'] == i)][target], s=300 \
                            , c='black')
            plt.xlabel(feature_column)
            plt.ylabel(target)
            plt.title('Cluster of ' + feature_product_seg + ' with ' + target)
            plt.legend()
            col2.pyplot()


        ## Visualization between the features each other
        for idx,val in enumerate(list_target):
            cluster_visual(column_label, k, feature_product, val)

    if col1.checkbox("Features after clustering"):

        col2.markdown('** The information of Features after Clustering **')
        col2.markdown('***')

        ## Box plot to show the different of the features in each cluster
        n_clusters = k


        x_data = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7']
        colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)',
                  'rgba(22, 80, 57, 0.5)', 'rgba(127, 65, 14, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)']

        cutoff_quantile = 95
        cl = 'clusters_' + str(n_clusters)
        for fild in range(0, len(product_column)):
            field_to_plot = product_column[fild]
            y_data = list()
            ymax = 0
            for i in np.arange(0, n_clusters):
                y0 = products_reordered[products_reordered[cl] == i][field_to_plot].values
                y0 = y0[y0 < np.percentile(y0, cutoff_quantile)]
                # if ymax < max(y0): ymax = max(y0)
                y_data.insert(i, y0)

            traces = []

            for xd, yd, cls in zip(x_data[:n_clusters], y_data, colors[:n_clusters]):
                traces.append(go.Box(y=yd, name=xd, boxpoints=False, jitter=0.5, whiskerwidth=0.2, fillcolor=cls,
                                     marker=dict(size=1, ),
                                     line=dict(width=1),
                                     ))

            layout = go.Layout(
                yaxis=dict(autorange=True, showgrid=True, zeroline=True,
                           dtick=int(ymax / 10),
                           gridcolor='white', gridwidth=0.1, zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2, ),
                margin=dict(l=40, r=30, b=50, t=50, ),
                paper_bgcolor='white',
                plot_bgcolor='rgba(93, 164, 214, 0.5)',
                showlegend=False
            )

            fig = go.Figure(data=traces, layout=layout)
            fig.update_layout(
                title= 'Difference in ' + field_to_plot + ' with ' + str(n_clusters) + ' Clusters.'
            )
            col2.plotly_chart(fig)








