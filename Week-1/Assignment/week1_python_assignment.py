# PROGRAM 01 | Cinema Ticket Calculator
# user inputs
user_age = int(input("Enter your age: "))
tickets = int(input("Tickets: "))
if user_age >= 12:
    charge = 300
else:
    charge = 180
# total bill
total_bill = charge * tickets
print(f"Total: Rs. {total_bill}")

print("-"*30)    
# PROGRAM 02 | Name Tag Generator
# user inputs 
first_name = input("Enter first name: ").strip().title().capitalize()
last_name = input("Enter last name: ").strip().title().capitalize()

full_name = first_name +' '+ last_name
title = first_name[0] + "." + last_name[0] + "."
count_letters = len(first_name)+len(last_name)
print(f"Full name: {full_name}\nTitle: {title}\nNumber of letters: {count_letters}")

print("-"*30)
# PROGRAM 03 | Five-Day Step Tracker
# empty list
steps = []
# taking user input from day 1 to 5
for days in range(1, 6):
    print("Day:", days)
    count = int(input("You have walked: "))
    steps.append(count)
print(f"Five days steps: {steps}")
# total steps, highest and average steps
total = sum(steps)
highest = max(steps)
average = total / 5
# count the qualifying steps
qualifying_steps = 0
for step in steps:
    if step >= 6000:
        qualifying_steps += 1
print(f"Qualifying days: {qualifying_steps}")

# Stationery Price Manager

Stationery = {
    "pen": 20,
    "notebook": 150,
    "easer": 5
}

print("-"*30)
# PROGRAM 04 | Stationery Price Manager
# user ask to update the item and its price
item = input("Enter item: ").lower()
price = int(input("Enter price: "))
# update the Stationery
if item in Stationery:
    Stationery.update({item: price})
else:
    print("Item not found.")
print("\nStationery Price List:")
# update name, price in Stationery
for name, price in Stationery.items():
    print(f"{name}: Rs. {price}")
# total prices
total = sum(Stationery.values())
print(f"Total Price:Rs. {total}")

print("-"*30)
# PROGRAM 05 | Three-Subject Result Checker
def average_marks(a: int, b: int, c: int) -> float:
    """average marks of three subjects."""
    return (a + b + c) / 3
# Assert Test
assert average_marks(60, 70, 80) == 70.0
# User Input
subject1 = int(input("Enter Subject 1 marks: "))
subject2 = int(input("Enter Subject 2 marks: "))
subject3 = int(input("Enter Subject 3 marks: "))
# Function Call
average = average_marks(subject1, subject2, subject3)
print("Average Marks:", average)
# Result
if average >= 50:
    print("Result: Pass")
else:
    print("Result: Needs improvement")

print("-"*30)
# PROGRAM 06 | Visitor Register
import json

# Create Movie Watchlist
movies = [
    {"title": "Inception", "watched": True},
    {"title": "Interstellar", "watched": False},
    {"title": "Avatar", "watched": False}
]

# Save Movies to JSON File
with open("watchlist.json", "w") as file:
    json.dump(movies, file, indent=4)

# Load Movies from JSON File
with open("watchlist.json", "r") as file:
    watchlist = json.load(file)

# Print Second Movie
print("Second Movie:", watchlist[1]["title"])

# Print Unwatched Movies
print("\nMovies Not Yet Watched:")

for movie in watchlist:
    if movie["watched"] == False:
        print(movie["title"])

print("-"*30)
# PROGRAM 07 | Movie Watchlist with JSON
import json
# Create Movie Watchlist
movies = [
    {"title": "Inception", "watched": True},
    {"title": "Interstellar", "watched": False},
    {"title": "Avatar", "watched": False}
]
# Save Movies to JSON File
with open("watchlist.json", "w") as file:
    json.dump(movies, file, indent=4)
# Load Movies from JSON File
with open("watchlist.json", "r") as file:
    watchlist = json.load(file)
# Print Second Movie
print("Second Movie:", watchlist[1]["title"])
# Print Unwatched Movies
print("\nMovies Not Yet Watched:")

for movie in watchlist:
    if movie["watched"] == False:
        print(movie["title"])

print("-"*30)
# Program 10 | Book Club Database

import sqlite3
# Create or Connect to Database
conn = sqlite3.connect("book_club.db")
cursor = conn.cursor()
# Create Members Table
cursor.execute("""
CREATE TABLE IF NOT EXISTS members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL
)
""")
# Insert Two Members
cursor.execute("INSERT INTO members (name) VALUES (?)", ("Ali",))
cursor.execute("INSERT INTO members (name) VALUES (?)", ("Ahmed",))
# Save Changes
conn.commit()
# Read Members
cursor.execute("SELECT name FROM members")
members = cursor.fetchall()
# Assert Test
assert len(members) >= 2
# Display Members
print("Book Club Members:")
for member in members:
    print(member[0])
# Close Database
conn.close()
