"""Menu-based task tracker depends on JSON"""

# importing library
import json
from pathlib import Path

# Making the path/file for the tasks in json format
TASKS_FILE = Path(__file__).with_name("tasks.json")

def load_tasks():
    """Load all saved tasks or start with an empty list."""
    try:
        with TASKS_FILE.open('r') as file:
            tasks = json.load(file)
        if not isinstance(tasks, list) or any(
            not isinstance(tasks, dict)
            or not isinstance(tasks.get("title"), str)
            or not isinstance(tasks.get("done"), bool)
            for task in tasks
        ):
            raise ValueError("Unexpected task data format")
        return tasks
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, ValueError):
        print("Could not read tasks.json. Starting with empty task list.")
        return []

def save_tasks(tasks):
    """Save the current task list to JSON"""
    with TASKS_FILE.open("w") as file:
        json.dump(tasks, file, indent=4)

def add_task(tasks):
    """Add new task"""
    title = input("Task Title: ").strip()
    if not title:
        print("Task title cannot be empty")
        return
    tasks.append({"title":title, "done": False})
    save_tasks(tasks)
    print("✅ Task added.")

# main function
def main():
    # load all present tasks
    tasks = load_tasks()
    # loop of main menu
    while True:
        print("="*52)
        print("\t\t    TASK TRACKER")
        print("="*52)
        print("1. Add   2. View   3. Complete   4. Delete   5. Exit")
        choice = int(input("Choice: "))
        if choice == 1:
            add_task(tasks)
        elif choice == 2:
            view_task(tasks)
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