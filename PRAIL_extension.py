import csv
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def read_predictions_csv(filename):
    today = datetime.today().date()
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            date = datetime.strptime(row['konec_datum_predviden'], '%Y-%m-%d').date()
            if date >= today:
                yield date, row['oprema_prikljucek_status_id']

def main():
    filename = 'predictions.csv'
    malfunction_dates = []
    oprema_status_ids = []
    total_malfunctioning = 0
    
    for date, oprema_status_id in read_predictions_csv(filename):
        # Only append if oprema_status_id is not 4, 15, or 16
        if oprema_status_id not in ['4', '15', '16']:
            malfunction_dates.append(date)
            oprema_status_ids.append(oprema_status_id)
            total_malfunctioning += 1
            
        if total_malfunctioning >= 500:
            print(f"Date when 500 or more devices will malfunction: {date.strftime('%d-%m-%Y')}")
            break
    
    else:
        print("Not enough data to reach 500 malfunctioning devices.")

    # Convert malfunction_dates and oprema_status_ids to DataFrame
    df = pd.DataFrame({'predviden_konec_datum': malfunction_dates, 'oprema_prikljucek_status_id': oprema_status_ids})
    

    # Plotting
    plt.figure(figsize=(10, 7))
    sns.set(style="darkgrid")
    sns.histplot(data=df, x='predviden_konec_datum', bins=10, kde=True)
    plt.title('Graf prikaza število naprav čez čas')
    plt.xlabel('Čas')
    plt.ylabel('Gostota')
    plt.xticks()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
