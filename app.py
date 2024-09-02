import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime, timedelta
import sqlite3
from sklearn.ensemble import RandomForestRegressor

# Gemini API configuration
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
API_KEY = ""

# Database setup
DB_NAME = "inventory.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS inventory
                 (id INTEGER PRIMARY KEY, 
                  item TEXT, 
                  quantity INTEGER, 
                  price REAL, 
                  category TEXT, 
                  supplier TEXT, 
                  reorder_point INTEGER,
                  last_updated DATE)''')
    c.execute('''CREATE TABLE IF NOT EXISTS orders
                 (id INTEGER PRIMARY KEY, 
                  item TEXT, 
                  quantity INTEGER, 
                  order_date DATE, 
                  expected_delivery DATE,
                  status TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS transactions
                 (id INTEGER PRIMARY KEY, 
                  item TEXT, 
                  quantity INTEGER, 
                  transaction_type TEXT, 
                  transaction_date DATE)''')
    c.execute('''CREATE TABLE IF NOT EXISTS suppliers
                 (id INTEGER PRIMARY KEY,
                  name TEXT,
                  contact_person TEXT,
                  email TEXT,
                  phone TEXT)''')
    conn.commit()
    conn.close()

def migrate_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Check if reorder_point column exists in inventory table
    c.execute("PRAGMA table_info(inventory)")
    columns = [column[1] for column in c.fetchall()]
    if 'reorder_point' not in columns:
        c.execute("ALTER TABLE inventory ADD COLUMN reorder_point INTEGER")
    
    # Check if expected_delivery column exists in orders table
    c.execute("PRAGMA table_info(orders)")
    columns = [column[1] for column in c.fetchall()]
    if 'expected_delivery' not in columns:
        c.execute("ALTER TABLE orders ADD COLUMN expected_delivery DATE")
    
    conn.commit()
    conn.close()

@st.cache_resource
def get_db_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def get_inventory():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM inventory", conn)
    return df

def get_orders():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM orders", conn)
    return df

def get_transactions():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM transactions", conn)
    return df

def get_suppliers():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM suppliers", conn)
    return df

def add_item(item, quantity, price, category, supplier, reorder_point):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        reorder_point = float(reorder_point) if reorder_point is not None else None
    except ValueError:
        reorder_point = None
    c.execute("INSERT INTO inventory (item, quantity, price, category, supplier, reorder_point, last_updated) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (item, quantity, price, category, supplier, reorder_point, datetime.now().date()))
    conn.commit()
    add_transaction(item, quantity, "Initial Stock")
    
def update_item(item, quantity, price):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE inventory SET quantity = ?, price = ?, last_updated = ? WHERE item = ?",
              (quantity, price, datetime.now().date(), item))
    conn.commit()
    add_transaction(item, quantity, "Update")

def add_order(item, quantity, status, expected_delivery):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO orders (item, quantity, order_date, expected_delivery, status) VALUES (?, ?, ?, ?, ?)",
              (item, quantity, datetime.now().date(), expected_delivery, status))
    conn.commit()
    add_transaction(item, -quantity, "Order")

def add_transaction(item, quantity, transaction_type):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO transactions (item, quantity, transaction_type, transaction_date) VALUES (?, ?, ?, ?)",
              (item, quantity, transaction_type, datetime.now().date()))
    conn.commit()

def add_supplier(name, contact_person, email, phone):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO suppliers (name, contact_person, email, phone) VALUES (?, ?, ?, ?)",
              (name, contact_person, email, phone))
    conn.commit()

def call_gemini_api(prompt):
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(f"{GEMINI_API_URL}?key={API_KEY}", headers=headers, json=data)
    return response.json()

def predict_stock(df, item, days):
    item_data = df[df['item'] == item].sort_values('last_updated')
    X = np.array(range(len(item_data))).reshape(-1, 1)
    y = item_data['quantity'].values
    model = RandomForestRegressor(n_estimators=100).fit(X, y)
    future_X = np.array(range(len(item_data), len(item_data) + days)).reshape(-1, 1)
    return model.predict(future_X)

def main():
    st.set_page_config(page_title="Advanced Inventory Management System", layout="wide")
    st.title('Advanced Inventory Management System')
    
    init_db()
    migrate_db()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Inventory Management", "Order Management",
                                      "Supplier Management", "Reports", "Predictive Analytics", "AI Assistant"])
    
    if page == "Dashboard":
        st.header('Inventory Dashboard')
        
        if st.button('Refresh Data'):
            st.rerun()
        
        df = get_inventory()
        orders_df = get_orders()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Items", df['item'].nunique())
        with col2:
            st.metric("Total Stock", df['quantity'].sum())
        with col3:
            st.metric("Total Value", f"${(df['quantity'] * df['price']).sum():.2f}")
        with col4:
            st.metric("Pending Orders", len(orders_df[orders_df['status'] == 'Pending']))
        
        # Recent transactions
        transactions_df = get_transactions().sort_values('transaction_date', ascending=False).head(10)
        st.subheader("Recent Transactions")
        st.dataframe(transactions_df)
        
        # Low stock alert
        if 'reorder_point' in df.columns:
            try:
                # Convert reorder_point to numeric, replacing any non-numeric values with NaN
                df['reorder_point'] = pd.to_numeric(df['reorder_point'], errors='coerce')
                
                # Filter out rows where reorder_point is not NaN and quantity is less than or equal to reorder_point
                low_stock = df[(df['reorder_point'].notna()) & (df['quantity'] <= df['reorder_point'])]
                
                if not low_stock.empty:
                    st.warning("Low Stock Alert!")
                    st.dataframe(low_stock)
                else:
                    st.info("All items are above their reorder points.")
            except Exception as e:
                st.error(f"An error occurred while checking low stock: {str(e)}")
                st.info("Please ensure all reorder point values are numeric.")
        else:
            st.info("Reorder point data is not available.")
        
        # Stock level visualization
        fig = px.bar(df, x='item', y='quantity', title='Current Stock Levels')
        st.plotly_chart(fig)
        
    elif page == "Inventory Management":
        st.header('Inventory Management')
        
        df = get_inventory()
        
        # Add new item
        st.subheader("Add New Item")
        new_item = st.text_input("Item Name")
        new_quantity = st.number_input("Quantity", min_value=0)
        new_price = st.number_input("Price", min_value=0.0, format="%.2f")
        new_category = st.text_input("Category")
        new_supplier = st.selectbox("Supplier", get_suppliers()['name'].unique())
        new_reorder_point = st.number_input("Reorder Point", min_value=0)
        if st.button("Add Item"):
            add_item(new_item, new_quantity, new_price, new_category, new_supplier, new_reorder_point)
            st.success("Item added successfully!")
            st.rerun()
        
        # Update existing item
        st.subheader("Update Existing Item")
        update_item_name = st.selectbox("Select Item to Update", df['item'].unique())
        update_quantity = st.number_input("New Quantity", min_value=0)
        update_price = st.number_input("New Price", min_value=0.0, format="%.2f")
        if st.button("Update Item"):
            update_item(update_item_name, update_quantity, update_price)
            st.success("Item updated successfully!")
            st.rerun()
        
        # Display current inventory
        st.subheader("Current Inventory")
        st.dataframe(df)
        
    elif page == "Order Management":
        st.header('Order Management')
        
        # New order form
        st.subheader('Create New Order')
        order_item = st.selectbox('Select Item', get_inventory()['item'].unique())
        order_quantity = st.number_input('Quantity', min_value=1)
        order_status = st.selectbox('Status', ['Pending', 'Shipped', 'Delivered'])
        expected_delivery = st.date_input("Expected Delivery Date")
        if st.button('Create Order'):
            add_order(order_item, order_quantity, order_status, expected_delivery)
            st.success('Order created successfully!')
            st.rerun()
        
        # Display orders
        orders_df = get_orders()
        st.subheader("Current Orders")
        st.dataframe(orders_df)
        
        # Order fulfillment
        st.subheader("Order Fulfillment")
        order_to_fulfill = st.selectbox("Select Order to Fulfill", orders_df[orders_df['status'] == 'Pending']['id'])
        if st.button("Fulfill Order"):
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("UPDATE orders SET status = 'Delivered' WHERE id = ?", (order_to_fulfill,))
            conn.commit()
            st.success("Order fulfilled successfully!")
            st.rerun()
        
    elif page == "Supplier Management":
        st.header('Supplier Management')
        
        # Add new supplier
        st.subheader("Add New Supplier")
        new_supplier_name = st.text_input("Supplier Name")
        new_contact_person = st.text_input("Contact Person")
        new_email = st.text_input("Email")
        new_phone = st.text_input("Phone")
        if st.button("Add Supplier"):
            add_supplier(new_supplier_name, new_contact_person, new_email, new_phone)
            st.success("Supplier added successfully!")
            st.rerun()
        
        # Display suppliers
        suppliers_df = get_suppliers()
        st.subheader("Current Suppliers")
        st.dataframe(suppliers_df)
        
        # Supplier performance
        st.subheader("Supplier Performance")
        selected_supplier = st.selectbox("Select Supplier", suppliers_df['name'])
        supplier_items = get_inventory()[get_inventory()['supplier'] == selected_supplier]
        st.write(f"Items supplied by {selected_supplier}:")
        st.dataframe(supplier_items)
        
    elif page == "Reports":
        st.header('Inventory Reports')
        
        report_type = st.selectbox("Select Report Type", ["Inventory Valuation", "Stock Movement", "Sales Report", "Reorder Report"])
        
        if report_type == "Inventory Valuation":
            df = get_inventory()
            df['total_value'] = df['quantity'] * df['price']
            st.dataframe(df[['item', 'quantity', 'price', 'total_value']])
            fig = px.bar(df, x='item', y='total_value', title='Inventory Valuation')
            st.plotly_chart(fig)
        
        elif report_type == "Stock Movement":
            transactions_df = get_transactions()
            fig = px.line(transactions_df, x='transaction_date', y='quantity', color='item', title='Stock Movement Over Time')
            st.plotly_chart(fig)
        
        elif report_type == "Sales Report":
            orders_df = get_orders()
            sales_df = orders_df[orders_df['status'] == 'Delivered']
            sales_by_item = sales_df.groupby('item')['quantity'].sum().reset_index()
            fig = px.pie(sales_by_item, values='quantity', names='item', title='Sales by Item')
            st.plotly_chart(fig)
        
        elif report_type == "Reorder Report":
            df = get_inventory()
            if 'reorder_point' in df.columns:
                reorder_df = df[df['quantity'] <= df['reorder_point']]
                st.write("Items that need reordering:")
                st.dataframe(reorder_df)
                fig = px.bar(reorder_df, x='item', y=['quantity', 'reorder_point'], title='Items Needing Reorder')
                st.plotly_chart(fig)
            else:
                st.write("Reorder point data is not available.")
        
    elif page == "Predictive Analytics":
        st.header('Predictive Analytics')
        
        df = get_inventory()
        item_to_predict = st.selectbox("Select item for prediction", df['item'].unique())
        days_to_predict = st.number_input("Number of days to predict", min_value=1, max_value=90, value=30)
        
        if st.button('Generate Prediction'):
            predicted_stock = predict_stock(df, item_to_predict, days_to_predict)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pd.date_range(start=datetime.now(), periods=days_to_predict),
                y=predicted_stock,
                mode='lines+markers',
                name='Predicted Stock'
            ))
            fig.update_layout(title=f'Stock Prediction for {item_to_predict}', xaxis_title='Date', yaxis_title='Predicted Quantity')
            st.plotly_chart(fig)
            
            # AI-powered inventory insights
            prompt = f"""Based on the stock prediction for {item_to_predict} over the next {days_to_predict} days:
            1. Analyze the trend and potential impact on inventory management
            2. Suggest optimal reorder timing and quantity
            3. Identify any potential risks or opportunities
            4. Recommend actions to optimize inventory levels

            Predicted stock levels: {predicted_stock.tolist()}"""
            response = call_gemini_api(prompt)
            st.subheader("AI-Powered Inventory Insights")
            st.write(response['candidates'][0]['content']['parts'][0]['text'])
        
    elif page == "AI Assistant":
        st.header('AI Inventory Assistant')
        
        user_input = st.text_input("Ask me anything about inventory management or for help with the application")
        if user_input:
            df = get_inventory()
            orders_df = get_orders()
            prompt = f"""You are an AI assistant for an advanced inventory management system. 
            The user has asked: "{user_input}"
            Provide a helpful, informative, and friendly response. If the query is about how to use the application, 
            provide step-by-step guidance. If it's a general inventory management question, provide best practices and advice.
            Include relevant data analysis or recommendations if applicable.

            Here's a summary of the current inventory state to inform your response:
            - Total number of items: {df['item'].nunique()}
            - Total stock: {df['quantity'].sum()}
            - Total inventory value: ${(df['quantity'] * df['price']).sum():.2f}
            - Number of suppliers: {df['supplier'].nunique()}
            - Items needing reorder: {len(df[df['quantity'] <= df['reorder_point']]) if 'reorder_point' in df.columns else 'N/A'}
            - Most valuable item: {df.loc[df['price'].idxmax(), 'item']} (${df['price'].max():.2f})
            - Item with highest stock: {df.loc[df['quantity'].idxmax(), 'item']} ({df['quantity'].max()} units)
            - Pending orders: {len(orders_df[orders_df['status'] == 'Pending'])}

            Based on this information and the user's query, provide a comprehensive response.
            """
            response = call_gemini_api(prompt)
            st.write(response['candidates'][0]['content']['parts'][0]['text'])
        
        # AI-powered inventory optimization suggestions
        if st.button("Get AI Inventory Optimization Suggestions"):
            df = get_inventory()
            orders_df = get_orders()
            transactions_df = get_transactions()
            
            prompt = f"""Analyze the current inventory state and provide optimization suggestions:
            1. Identify items that may be overstocked or understocked
            2. Suggest improvements to the reorder points
            3. Analyze the sales trends and recommend inventory adjustments
            4. Propose strategies to reduce holding costs while maintaining adequate stock levels
            5. Identify any potential supply chain risks based on the current inventory and order data

            Inventory summary:
            {df.to_string()}

            Recent orders:
            {orders_df.tail(10).to_string()}

            Recent transactions:
            {transactions_df.tail(20).to_string()}

            Provide a detailed analysis and actionable recommendations.
            """
            response = call_gemini_api(prompt)
            st.subheader("AI Inventory Optimization Suggestions")
            st.write(response['candidates'][0]['content']['parts'][0]['text'])
        
        # AI-powered demand forecasting
        if st.button("Generate AI Demand Forecast"):
            df = get_inventory()
            transactions_df = get_transactions()
            
            prompt = f"""Based on the historical transaction data and current inventory levels, 
            generate a demand forecast for the next 30 days:
            1. Predict the expected demand for each item
            2. Identify any seasonal trends or patterns
            3. Suggest inventory levels to meet the forecasted demand
            4. Highlight any items that may require special attention due to changing demand patterns

            Current inventory:
            {df.to_string()}

            Transaction history:
            {transactions_df.to_string()}

            Provide a comprehensive demand forecast and inventory recommendations.
            """
            response = call_gemini_api(prompt)
            st.subheader("AI-Generated Demand Forecast")
            st.write(response['candidates'][0]['content']['parts'][0]['text'])
        
        # AI-powered inventory health check
        if st.button("Perform AI Inventory Health Check"):
            df = get_inventory()
            orders_df = get_orders()
            suppliers_df = get_suppliers()
            
            prompt = f"""Conduct a comprehensive inventory health check and provide insights:
            1. Assess the overall health of the inventory
            2. Identify any potential issues or inefficiencies
            3. Evaluate the balance between stock levels and demand
            4. Analyze supplier performance and diversity
            5. Suggest improvements for inventory management practices

            Inventory data:
            {df.to_string()}

            Order data:
            {orders_df.to_string()}

            Supplier data:
            {suppliers_df.to_string()}

            Provide a detailed health check report with actionable recommendations.
            """
            response = call_gemini_api(prompt)
            st.subheader("AI Inventory Health Check Report")
            st.write(response['candidates'][0]['content']['parts'][0]['text'])

if __name__ == '__main__':
    main()
