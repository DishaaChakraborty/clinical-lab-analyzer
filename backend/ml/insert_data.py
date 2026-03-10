import pandas as pd
import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
print ("PASSWORD:", os.getenv("DATABASE_PASSWORD"))

def connect_to_database():
    """Connect to MySQL database"""
    try:
        connection = mysql.connector.connect(
            host=os.getenv('DATABASE_HOST', 'localhost'),
            user=os.getenv('DATABASE_USER', 'root'),
            password=os.getenv("DATABASE_PASSWORD"),
            database=os.getenv( "DATABASE_NAME")
        )
        return connection
    except Error as e:
        print(f"Error: {e}")
        return None

def insert_test_patients_from_csv(csv_path: str = 'backend/ml/data/synthetic_data.csv'):
    """Insert patient data from CSV into database"""
    
    # Load CSV
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} records")
    
    # Connect to database
    print("\nConnecting to MySQL database...")
    connection = connect_to_database()
    
    if not connection:
        print("❌ Failed to connect to database!")
        return
    
    cursor = connection.cursor()
    
    try:
        print("Inserting patient records...")
        
        # Get unique patients
        unique_patients = df.drop_duplicates(subset=['patient_id'])[
            ['patient_id', 'age', 'gender']
        ].head(50)  # Insert only first 50 patients
        
        inserted_count = 0
        
        for _, row in unique_patients.iterrows():
            try:
                sql = """
                INSERT INTO patients (name, age, gender, email, phone, is_active)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                
                # Generate sample data
                name = f"Patient {int(row['patient_id'])}"
                age = int(row['age'])
                gender = row['gender']
                email = f"patient{int(row['patient_id'])}@example.com"
                phone = f"98765{int(row['patient_id']):05d}"
                
                cursor.execute(sql, (name, age, gender, email, phone, True))
                inserted_count += 1
                
                if inserted_count % 10 == 0:
                    print(f"  ✓ Inserted {inserted_count} patients...")
                
            except Error as e:
                print(f"Error inserting patient {row['patient_id']}: {e}")
        
        connection.commit()
        print(f"\n✓ Successfully inserted {inserted_count} patients")
        
        # Insert sample reports for first 5 patients
        print("\nInserting sample reports...")
        
        report_count = 0
        for patient_id in range(1, 6):  # First 5 patients
            for report_num in range(1, 4):  # 3 reports each
                try:
                    sql = """
                    INSERT INTO reports (patient_id, report_date, file_name, file_type, is_parsed)
                    VALUES (%s, NOW(), %s, %s, %s)
                    """
                    
                    file_name = f"report_{patient_id}_{report_num}.csv"
                    
                    cursor.execute(sql, (patient_id, file_name, 'csv', True))
                    report_count += 1
                    
                except Error as e:
                    print(f"Error inserting report for patient {patient_id}: {e}")
        
        connection.commit()
        print(f"✓ Successfully inserted {report_count} reports")
        
        # Verify data
        print("\n=== VERIFYING DATA ===")
        
        # Count patients
        cursor.execute("SELECT COUNT(*) FROM patients;")
        patient_count = cursor.fetchone()[0]
        print(f"\nTotal patients in database: {patient_count}")
        
        # Count reports
        cursor.execute("SELECT COUNT(*) FROM reports;")
        report_count = cursor.fetchone()[0]
        print(f"Total reports in database: {report_count}")
        
        # Show sample patients
        print("\n=== SAMPLE PATIENTS ===")
        cursor.execute("SELECT id, name, age, gender, email FROM patients LIMIT 5;")
        for row in cursor.fetchall():
            print(f"ID: {row[0]}, Name: {row[1]}, Age: {row[2]}, Gender: {row[3]}, Email: {row[4]}")
        
        # Show sample reports
        print("\n=== SAMPLE REPORTS ===")
        cursor.execute("SELECT id, patient_id, file_name, created_at FROM reports LIMIT 5;")
        for row in cursor.fetchall():
            print(f"ID: {row[0]}, Patient: {row[1]}, File: {row[2]}, Created: {row[3]}")
        
        print("\n✓ Data insertion complete!")
        
    except Error as e:
        print(f"❌ Error: {e}")
        connection.rollback()
    
    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    insert_test_patients_from_csv()