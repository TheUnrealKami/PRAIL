import pyodbc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import joblib
import random

def load_data():
    try:
        print("Connecting to SQL Server...")
        # Connect to SQL Server
        conn = pyodbc.connect('DRIVER={SQL Server};SERVER=172.16.1.115;DATABASE=BI;Trusted_Connection=yes', timeout=30)
        
        # Load data from SQL Server
        query = """SELECT
                    op.oprema_sn,
                    DENSE_RANK() OVER (ORDER BY op.oprema_sn) AS assigned_sim_num,
                    op.ackey,
                    op.ident,
                    DENSE_RANK() OVER (ORDER BY op.ident) AS assigned_ident_num,
                    opz.oprema_prikljucek_status_id,
                    FLOOR(DATEDIFF(DAY,0, op.oprema_vpis_datum)) as oprema_vpis_datum,
                    FLOOR(DATEDIFF(DAY, 0,  opz.zacetek_datum)) as zacetek_datum,
                    FLOOR(DATEDIFF(DAY, op.oprema_vpis_datum, opz.zacetek_datum)) as Razlika,
                    opz.konec_datum,
                    op.prikljucek_id
                    FROM dbo.oprema_prikljucek_zgodovina opz
                    JOIN dbo.oprema_prikljucek op ON op.oprema_prikljucek_id = opz.oprema_prikljucek_id
                    JOIN dbo.oprema_prikljucek_status ops ON ops.oprema_prikljucek_status_id = opz.oprema_prikljucek_status_id
                    LEFT JOIN dbo.v_prikljucek p ON p.prikljucek_id = opz.prikljucek_id
                    WHERE oprema_sn != ''
                    AND opz.zacetek_datum > op.oprema_vpis_datum
                    AND op.ident like 'O[XC]IAD-INNBOXV60'"""
        print("Loading data from SQL Server...")
        data = pd.read_sql(query, conn)

        print(f"Loaded {data.shape[0]} rows of data")

        return data
    except pyodbc.Error as e:
        print("Error connecting to SQL Server:", e)
        return None

def train_random_forest(data):
    models = {}
    for sim_num, sim_num_group in data.groupby('assigned_sim_num'):
        for ident_num, ident_num_group in sim_num_group.groupby('assigned_ident_num'):
            if len(ident_num_group) < 2:  # Check if there are enough samples to split
                print(f"Not enough samples for sim_num. Making simple calculation....")
                data.loc[ident_num_group.index, 'konec_datum_predviden'] = ident_num_group['oprema_vpis_datum'] + ident_num_group['Razlika']
                continue

            # Extract features and target
            X = ident_num_group[['Razlika', 'oprema_vpis_datum']]  # Use double square brackets to create a DataFrame
            y = ident_num_group['zacetek_datum']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train Random Forest Regressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)

            # Fit the model
            model.fit(X_train, y_train)

            # Evaluate model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print(f"Random Forest for sim_num {sim_num}, ident_num {ident_num} - MSE: {mse}")

            models[(sim_num, ident_num)] = model

    # Save the model
    print("Creating and saving the model...")
    model_save_path = f"model.joblib"
    joblib.dump(models, model_save_path)

    return models

def make_prediction(models, assigned_sim_num, assigned_ident_num, Razlika, oprema_vpis_datum, data):
    model = models.get((assigned_sim_num, assigned_ident_num))
    if model:
        prediction = model.predict([[Razlika, oprema_vpis_datum]])
        if prediction < 45290:
            prediction += random.randint(1461, 2191)
        return prediction
    else:
        # Check if there are enough samples for the given sim_num
        sim_group = data[data['assigned_sim_num'] == assigned_sim_num]
        if len(sim_group) < 2:
            # Fallback to previous calculation
            prediction = oprema_vpis_datum + Razlika + random.randint(1461, 2191)
            return prediction
        else:
            return None

def predict_and_write_to_csv(data, models, output_file):
    try:
        print("Writing data to CSV file...")

        # Iterate over each row in the DataFrame
        rows_to_write = []  # List to store rows for which predictions are made
        for index, row in data.iterrows():
            assigned_sim_num = row['assigned_sim_num']
            assigned_ident_num = row['assigned_ident_num']
            razlika = row['Razlika']
            oprema_vpis_datum = row['oprema_vpis_datum']

            # Make prediction
            prediction = make_prediction(models, assigned_sim_num, assigned_ident_num, razlika, oprema_vpis_datum, data)
            
            # Append row and prediction to list
            row['konec_datum_predviden'] = prediction
            rows_to_write.append(row)

        # Convert rows to DataFrame
        rows_df = pd.DataFrame(rows_to_write)

        # Convert konec_datum_predviden from days to timestamp
        rows_df['konec_datum_predviden'] = rows_df['konec_datum_predviden'].apply(lambda x: datetime(1900, 1, 1) + timedelta(days=int(x)) if x != "FAILED" else None)
        rows_df['oprema_vpis_datum'] = rows_df['oprema_vpis_datum'].apply(lambda x: datetime(1900, 1, 1) + timedelta(days=int(x)) if x != "FAILED" else None)
        rows_df['zacetek_datum'] = rows_df['zacetek_datum'].apply(lambda x: datetime(1900, 1, 1) + timedelta(days=int(x)) if x != "FAILED" else None)
        
        # Write DataFrame to CSV
        rows_df.to_csv(output_file, index=False)

        print("Data successfully written to CSV file.")

    except Exception as e:
        print("Error writing data to CSV file:", e)

# Saving
data = load_data()
if data is not None:
    # Train Random Forest Regressor
    models = train_random_forest(data)

    # Write predictions to CSV
    output_file = "predictions.csv"
    predict_and_write_to_csv(data, models, output_file)
