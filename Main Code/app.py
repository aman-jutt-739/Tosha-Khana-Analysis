import flask
from flask import Flask, render_template
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive Agg backend
import matplotlib.pyplot as plt
from matplotlib import gridspec
import io
import base64
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

app = Flask(__name__)
# In-memory cache for plots
plot_cache = {}
# Load and preprocess dataset (exact notebook code)
def load_data():
    try:
        df = pd.read_csv('static\\Refined_TK_dataver2.csv')
        df['Affiliation'] = df['Affiliation'].fillna('Unknown')
        df['Affiliation'] = df['Affiliation'].replace({
            'Gen Mus': 'Gen. Musharraf',
            'Gen Mush': 'Gen. Musharraf',
            'Gen. Musharrafarraf': 'Gen. Musharraf'
        })
        df['Detail of Gifts'] = df['Detail of Gifts'].replace({'One Carpet': 'One carpet'})
        pd.options.display.float_format = '{:,.0f}'.format
        df['Assessed Value'] = pd.to_numeric(df['Assessed Value'], errors='coerce')
        df['Retention Cost'] = pd.to_numeric(df['Retention Cost'], errors='coerce')
        df.dropna(subset=['Assessed Value', 'Retention Cost'], how='all', inplace=True)
        return df, None
    except FileNotFoundError:
        return None, "Error: 'Refined_TK_dataver2.csv' not found in the project directory."
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}"

# Helper function to convert plot to base64
def plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)  # Reduced DPI for performance
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close('all')  # Close all figures to free memory
    return img_str
# Compute probability distributions
def compute_probabilities(df):
    # Probability of retention
    retained_prob = df['Retained'].value_counts(normalize=True).get('Yes', 0)
    
    # Set pandas display format
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    # Conditional probability of non-retention by Affiliation
    affiliation_group = df.groupby('Affiliation')['Retained'].value_counts(normalize=True).unstack()
    affiliation_probs = affiliation_group['No'].dropna().sort_values(ascending=False).to_dict()
    
    # Posterior probabilities of non-retention by Item Category
    category_group = df.groupby('Item Category')['Retained'].value_counts(normalize=True).unstack()
    category_probs = category_group['No'].dropna().sort_values(ascending=False).to_dict()
    
    return retained_prob, affiliation_probs, category_probs
# Generate all seven graphs
def generate_graphs(df):
    cache_key = 'graphs'
    if cache_key in plot_cache:
        return plot_cache[cache_key]
    
    plots = []
    
    try:
        # Graph 1 & 2: Main and Inset Histograms
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax_main = plt.subplot(gs[0])
        ax_inset = plt.subplot(gs[1])
        ax_main.hist(df['Assessed Value'].dropna(), bins=100, color='skyblue', edgecolor='black')
        ax_main.set_title("Full Distribution (Compressed)")
        ax_main.set_xlabel("Assessed Value (PKR)")
        ax_main.set_ylabel("Frequency")
        assessed_99 = df['Assessed Value'].quantile(0.90)
        filtered = df[df['Assessed Value'] <= assessed_99]
        ax_inset.hist(filtered['Assessed Value'], bins=50, color='orange', edgecolor='black')
        ax_inset.set_title("Zoomed-in (0–90%)")
        ax_inset.set_xlabel("Assessed Value")
        plt.tight_layout()
        plots.append({'title': 'Assessed Value Histograms', 'image': plot_to_base64()})

        # Graph 3: Item Category Bar Plot
        plt.figure(figsize=(10, 6))
        df['Item Category'].value_counts().plot(kind='bar', color='lightgreen', edgecolor='black')
        plt.title('Item Category Distribution')
        plt.xlabel('Item Category')
        plt.ylabel('Count')
        plots.append({'title': 'Item Category Distribution', 'image': plot_to_base64()})

        # Graph 4: Affiliation Bar Plot
        plt.figure(figsize=(10, 6))
        df['Affiliation'].value_counts().plot(kind='bar', color='lightcoral', edgecolor='black')
        plt.title('Affiliation Distribution')
        plt.xlabel('Affiliation')
        plt.ylabel('Count')
        plots.append({'title': 'Affiliation Distribution', 'image': plot_to_base64()})

        # Graph 5: Assessed Value Over Time
        df_time = df.copy()
        df_time['Date'] = pd.to_datetime(df_time['Date'], errors='coerce')
        df_time = df_time.dropna(subset=['Date', 'Assessed Value'])
        df_time = df_time.sort_values('Date')
        plt.figure(figsize=(10, 6))
        plt.plot(df_time['Date'], df_time['Assessed Value'], marker='o', linestyle='-', color='b')
        plt.ticklabel_format(style='plain', axis='y')
        plt.title('Assessed Value of Gifts Over Time')
        plt.xlabel('Date')
        plt.ylabel('Assessed Value')
        plt.grid(True)
        plots.append({'title': 'Assessed Value of Gifts Over Time', 'image': plot_to_base64()})

        # Graph 6: Pie Chart of Selected Affiliations
        filtered_df = df[df['Affiliation'].isin(['PTI', 'PMLN', 'PPP', 'Bureaucracy', 'Media', 'Police', 'Military', 'Gen. Musharraf'])]
        affiliation_counts = filtered_df['Affiliation'].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(affiliation_counts, labels=affiliation_counts.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999', '#66b3ff'])
        plt.title('Number of Gifts Received')
        plots.append({'title': 'Number of Gifts Received', 'image': plot_to_base64()})

        # Graph 7: Pie Chart of PTI and PMLN
        filtered_df = df[df['Affiliation'].isin(['PTI', 'PMLN'])]
        affiliation_counts = filtered_df['Affiliation'].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(affiliation_counts, labels=affiliation_counts.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999', '#66b3ff'])
        plt.title('Number of Gifts Received During PTI and PMLN Eras')
        plots.append({'title': 'Number of Gifts Received During PTI and PMLN Eras', 'image': plot_to_base64()})

        plot_cache[cache_key] = plots
        return plots
    except Exception as e:
        return [{'title': 'Error', 'image': None, 'error': f"Error generating graphs: {str(e)}"}]
# Compute linear regression
def compute_regression(df):
    cache_key = 'regression'
    if cache_key in plot_cache:
        return plot_cache[cache_key]
    
    try:
        # Drop rows with missing values
        df_reg = df.dropna(subset=['Assessed Value', 'Retention Cost'])
        
        # Check if enough data remains
        if len(df_reg) < 2:
            return None, None, "Error: Insufficient data for regression after dropping missing values."
        
        # Prepare data
        X = df_reg[['Assessed Value']]  # 2D array
        y = df_reg['Retention Cost']    # 1D array
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict
        y_pred = model.predict(X)
        
        # Generate plot
        plt.figure(figsize=(8, 5))
        plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')
        plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
        plt.title("Linear Regression: Retention Cost vs Assessed Value")
        plt.xlabel("Assessed Value (PKR)")
        plt.ylabel("Retention Cost (PKR)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Convert plot to base64
        img_str = plot_to_base64()
        
        # Model statistics
        stats = {
            'intercept': model.intercept_,
            'slope': model.coef_[0],
            'r2': r2_score(y, y_pred)
        }
        
        result = (img_str, stats, None)
        plot_cache[cache_key] = result
        return result
    except Exception as e:
        return None, None, f"Error in regression modeling: {str(e)}"

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/stats')
def stats():
    df, error = load_data()
    if error:
        return render_template('stats.html', error=error)
    stats_df = df[['Assessed Value', 'Retention Cost']].describe()
    return render_template('stats.html', stats=stats_df.to_dict())

@app.route('/graphs')
def graphs():
    df, error = load_data()
    if error:
        return render_template('graphs.html', error=error)
    
    plots = generate_graphs(df)
    if plots and 'error' in plots[0]:
        return render_template('graphs.html', error=plots[0]['error'])
    
    return render_template('graphs.html', plots=plots)

@app.route('/distributions')
def distributions():
    df, error = load_data()
    if error:
        return render_template('distributions.html', error=error)
    
    retained_prob, affiliation_probs, category_probs = compute_probabilities(df)
    return render_template('distributions.html', 
                         retained_prob=retained_prob, 
                         affiliation_probs=affiliation_probs, 
                         category_probs=category_probs)
@app.route('/regression')
def regression():
    df, error = load_data()
    if error:
        return render_template('regression.html', error=error)
    
    plot, stats, error = compute_regression(df)
    if error:
        return render_template('regression.html', error=error)
    
    return render_template('regression.html', plot=plot, stats=stats)
if __name__ == '__main__':
    app.run(debug=True, threaded=False)  # Single-threaded mode