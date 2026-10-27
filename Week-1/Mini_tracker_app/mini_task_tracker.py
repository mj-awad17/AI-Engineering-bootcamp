"""Menu-based task tracker depends on JSON"""

# importing library
import json

# main function
def main():
    
    # loop of main menu
    while True:
        print("="*28)
        print("\tTASK TRACKER")
        print("="*28)
        print("1. Add   2. View   3. Complete   4. Delete   5. Exit")
        choice = int(input("Choice: "))
        if choice == 1:
            print("add task")
        elif choice == 2:
            print("View task")
        elif choice == 3:
            print("Completed task")
        elif choice == 4:
            print("Delete task")
        elif choice == 5:
            print("Goodbye!")
            break
        else:
            print("Invalid choice.\nPlease select 1 to 5.")

if __name__ == "__main__":
    main()