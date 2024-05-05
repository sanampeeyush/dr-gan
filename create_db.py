import sqlite3

# Connect to SQLite database (creates a new file if not exists)
conn = sqlite3.connect('users.db')

# Create a cursor object to execute SQL commands
cursor = conn.cursor()

# Create users table
cursor.execute(
    '''
        CREATE TABLE IF NOT EXISTS users
        (
            id INTEGER PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    '''
)

# Hardcoded user data
users_data = [
    ('user@gmail.com', '12345678'),
    # ('user2@example.com', 'password2')
]

# Insert hardcoded users into the table
cursor.executemany(
    '''INSERT INTO users (email, password) VALUES (?, ?)''', users_data
)

# Commit changes and close the connection
conn.commit()
conn.close()

print("Database created and users added successfully.")
