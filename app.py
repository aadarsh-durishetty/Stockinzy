from flask import Flask, render_template, request, redirect, url_for, flash
from flask_bcrypt import Bcrypt
from flask_pymongo import PyMongo
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from bson.objectid import ObjectId
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import redis
import logging
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['MONGO_URI'] = "mongodb://localhost:27017/stockinzy"
app.config['SECRET_KEY'] = 'your_secret_key'

mongo = PyMongo(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
logging.basicConfig(level=logging.INFO)
# Redis configuration
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# Initialize Flask-SocketIO with Redis as the message queue
socketio = SocketIO(app, message_queue='redis://')

# Load and preprocess stock data
df = pd.read_csv('stockdata.csv')
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['is_quarter_end'] = (df['month'] % 3 == 0)
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

features = df[['open', 'high', 'low', 'close', 'volume', 'day', 'month', 'year', 'is_quarter_end']]
target = df['target']

imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_imputed)

X_train, X_valid, Y_train, Y_valid = train_test_split(features_scaled, target, test_size=0.1, random_state=2022)

model = XGBClassifier()
model.fit(X_train, Y_train)

# User Model
class User(UserMixin):
    def __init__(self, id, email, username, password):
        self.id = id
        self.email = email
        self.username = username
        self.password = password

@login_manager.user_loader
def load_user(user_id):
    user = mongo.db.users.find_one({"_id": ObjectId(user_id)})
    if not user:
        return None
    return User(str(user['_id']), user['email'], user['username'], user['password'])

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        username = request.form['username']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')

        user = mongo.db.users.find_one({"email": email})
        if user:
            flash('Email address already exists')
            return redirect(url_for('register'))

        mongo.db.users.insert_one({'email': email, 'username': username, 'password': password})
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = mongo.db.users.find_one({"username": username})
        if user and bcrypt.check_password_hash(user['password'], password):
            login_user(User(str(user['_id']), user['email'], user['username'], user['password']))
            return redirect(url_for('page1'))
        else:
            flash('Login Unsuccessful. Please check username and password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/page1')
@login_required
def page1():
    return render_template('page1.html')

@app.route('/overview')
@login_required
def overview():
    stock_symbols = df['stock_symbol'].unique()
    stocks = [{'stock_symbol': symbol} for symbol in stock_symbols]

    # Define sizes for each stock image
    image_sizes = {
        'AAPL': {'width': 200, 'height': 200},
        'ADBE': {'width': 250, 'height': 250},
        'AMZN': {'width': 220, 'height': 220},
        'CRM': {'width': 300, 'height': 200},
        'CSCO': {'width': 300, 'height': 200},
        'GOOGL': {'width': 200, 'height': 200},
        'IBM': {'width': 600, 'height': 200},
        'INTC': {'width': 300, 'height': 200},
        'META': {'width': 300, 'height': 200},
        'MSFT': {'width': 200, 'height': 200},
        'NFLX': {'width': 150, 'height': 250},
        'NVDA': {'width': 400, 'height': 300},
        'ORCL': {'width': 250, 'height': 250},
        'TSLA': {'width': 250, 'height': 250}
    }

    return render_template('overview.html', stocks=stocks, image_sizes=image_sizes)

@app.route('/stock_graph/<stock_symbol>')
@login_required
def stock_graph(stock_symbol):
    # Filter data for the selected stock symbol
    stock_data = df[df['stock_symbol'] == stock_symbol]

    # Plot the graph
    plt.figure(figsize=(10, 5))
    plt.plot(stock_data['date'], stock_data['close'], marker='o')
    plt.title(f'Stock Prices for {stock_symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price USD')
    plt.grid(True)

    # Save the plot to a bytes object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('stock_graph.html', plot_url=plot_url, stock_symbol=stock_symbol)

@app.route('/analysis', methods=['GET', 'POST'])
@login_required
def analysis():
    if request.method == 'POST':
        date = request.form.get('date')
        stock_symbol = request.form.get('stock_symbol')
        cache_key = f"{stock_symbol}_{date}"

        # Check if the result is cached in Redis
        cached_result = redis_client.get(cache_key)
        if cached_result:
            logging.info(f"Cache hit for key: {cache_key}")
            result = cached_result.decode('utf-8').split(',')
            prediction_text, latest_price, action_message, profit_message, profit_color = result
        else:
            logging.info(f"Cache miss for key: {cache_key}")
            # Prepare the input features for prediction
            stock_data = df[df['stock_symbol'] == stock_symbol]
            if stock_data.empty:
                flash('Stock symbol not found.')
                return redirect(url_for('analysis'))

            stock_data = stock_data.sort_values(by='date')

            try:
                analysis_date = pd.to_datetime(date, format='%d-%m-%Y')
            except ValueError:
                flash('Invalid date format.')
                return redirect(url_for('analysis'))

            if analysis_date > pd.Timestamp('2022-12-31'):
                # Future date
                last_entry = stock_data.iloc[-1]
                day_diff = (analysis_date - last_entry['date']).days
                predicted_price = last_entry['close'] * (1 + 0.0005 * day_diff)  # Use a reasonable growth rate

                prediction_text = "Stock price is predicted to increase."
                expected_change_percentage = ((predicted_price - last_entry['close']) / last_entry['close']) * 100
                action_message = "Consider buying the stock."
                profit_message = f"Expected profit: {expected_change_percentage:.2f}%"
                profit_color = 'green'
            elif analysis_date < pd.Timestamp('2010-01-01'):
                # Historical date
                first_entry = stock_data.iloc[0]
                day_diff = (first_entry['date'] - analysis_date).days
                predicted_price = first_entry['close'] * (1 - 0.0003 * day_diff)

                prediction_text = "Stock price analysis before available data."
                expected_change_percentage = ((first_entry['close'] - predicted_price) / first_entry['close']) * 100
                if expected_change_percentage > 50:
                    expected_change_percentage = 50  # Cap at 50% loss
                elif expected_change_percentage == 100:
                    expected_change_percentage = np.random.uniform(1, 50)  # Random value below 50%
                action_message = "Historical analysis."
                profit_message = f"Estimated decrease: {expected_change_percentage:.2f}%"
                profit_color = 'red'
            else:
                # Available date (2010-2022)
                relevant_data = stock_data[stock_data['date'] <= analysis_date]
                if relevant_data.empty:
                    last_entry = stock_data.iloc[0]
                else:
                    last_entry = relevant_data.iloc[-1]

                predicted_price = model.predict([[
                    last_entry['open'], last_entry['high'], last_entry['low'], last_entry['close'],
                    last_entry['volume'], last_entry['day'], last_entry['month'], last_entry['year'],
                    last_entry['is_quarter_end']
                ]])[0]

                if analysis_date < last_entry['date']:
                    prediction_text = "Stock price has decreased since the selected date."
                    expected_change_percentage = ((last_entry['close'] - predicted_price) / last_entry['close']) * 100
                    if expected_change_percentage > 50:
                        expected_change_percentage = 50  # Cap at 50% loss
                    elif expected_change_percentage == 100:
                        expected_change_percentage = np.random.uniform(1, 50)  # Random value below 50%
                    action_message = "Historical analysis."
                    profit_message = f"Decrease observed: {expected_change_percentage:.2f}%"
                    profit_color = 'red'
                else:
                    prediction_text = "Stock price prediction based on data."
                    expected_change_percentage = ((predicted_price - last_entry['close']) / last_entry['close']) * 100
                    if expected_change_percentage > 0:
                        action_message = "Consider the stock based on your risk appetite."
                        profit_message = f"Predicted change: {expected_change_percentage:.2f}%"
                        profit_color = 'green'
                    else:
                        expected_change_percentage = ((predicted_price - last_entry['close'])/ last_entry['open']) * last_entry['close']  # Ensure positive value for message
                        action_message = "Consider selling the stock."
                        profit_message = f"Predicted decrease: {expected_change_percentage:.2f}%"
                        profit_color = 'red'

            latest_price = last_entry['close']

            # Cache the result in Redis
            redis_client.setex(cache_key, 3600, ','.join([
                prediction_text, str(latest_price), action_message, profit_message, profit_color
            ]))

        return render_template('result.html', 
                               prediction=prediction_text, 
                               latest_price=latest_price, 
                               action=action_message, 
                               profit=profit_message,
                               profit_color=profit_color)

    return render_template('analysis.html')

@app.route('/buy_sell', methods=['GET', 'POST'])
@login_required
def buy_sell():
    if request.method == 'POST':
        action = request.form['action']
        stock_symbol = request.form['stock']
        stock = mongo.db.stocks.find_one({'user_id': current_user.id, 'stock_symbol': stock_symbol})
        if action == 'Buy':
            if stock:
                mongo.db.stocks.update_one({'user_id': current_user.id, 'stock_symbol': stock_symbol}, {'$inc': {'owned': 1}})
            else:
                mongo.db.stocks.insert_one({'user_id': current_user.id, 'stock_symbol': stock_symbol, 'owned': 1})
        elif action == 'Sell' and stock and stock['owned'] > 0:
            mongo.db.stocks.update_one({'user_id': current_user.id, 'stock_symbol': stock_symbol}, {'$inc': {'owned': -1}})
        
        mongo.db.history.insert_one({
            'user_id': current_user.id,
            'action': action,
            'stock_symbol': stock_symbol,
            'timestamp': datetime.utcnow()
        })

        # Emit real-time update to clients
        socketio.emit('update', {'action': action, 'stock_symbol': stock_symbol})

        flash(f"{action} action recorded for {stock_symbol}.")
        return redirect(url_for('buy_sell'))
    
    stocks = df['stock_symbol'].unique()
    stock_data = []
    for symbol in stocks:
        if 'company' in df.columns:
            company = df[df['stock_symbol'] == symbol]['company'].iloc[0]
        else:
            company = symbol  # Use stock_symbol as the company name if not found
        owned = mongo.db.stocks.find_one({'user_id': current_user.id, 'stock_symbol': symbol})
        stock_data.append({'stock_symbol': symbol, 'company': company, 'owned': owned['owned'] if owned else 0})

    return render_template('buy_sell.html', stocks=stock_data)

@app.route('/history')
@login_required
def history():
    actions = mongo.db.history.find({'user_id': current_user.id})
    action_list = [
        {
            'action': a['action'],
            'stock_symbol': a.get('stock_symbol', 'N/A'),
            'quantity': a.get('quantity', 1),  # Assuming 1 if not provided
            'timestamp': a['timestamp']
        } 
        for a in actions
    ]
    return render_template('history.html', actions=action_list)

@app.route('/')
def index():
    return redirect(url_for('login'))

if __name__ == '__main__':
    socketio.run(app, debug=True)
