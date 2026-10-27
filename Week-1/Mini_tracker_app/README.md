# Menu-Based Task Tracker 

A simple, menu-based task tracker built with core Python. It lets users add, view, complete, and delete tasks. Tasks are stored in a JSON file so they can be loaded again when the program runs.

## Features

- **Add tasks:** Enter a task title to add it with a `Pending` status.
- **View tasks:** Display all tasks with their task numbers and statuses.
- **Complete tasks:** Select a task by number and mark it as `Done`.
- **Delete tasks:** Select a task by number and remove it.
- **Save tasks:** Store tasks in `tasks.json` using JSON.
- **Load tasks:** Load saved tasks when the program starts.
- **Input validation:** Handle empty task titles, non-numeric task numbers, and invalid task selections.
- **File error handling:** Start with an empty task list if `tasks.json` is missing or contains invalid data.

## Requirements

- Python 3.7 or later
- No external packages are required; the project uses Python's built-in `json` and `pathlib` modules.

## Project Files

```text
mini_task-tracker_app/
├── mini_task_tracker.py
└── tasks.json              # Created automatically when tasks are saved
```

The program stores `tasks.json` in the same directory as the Python script.

## How to Run

1. Save the Python code in a file named `mini_task_tracker.py`.
2. Open a terminal in that file's directory.
3. Run:

   ```bash
   python mini_task_tracker.py
   ```

4. Use the menu to manage your tasks.

## Menu Options

```text
====================================================
                    TASK TRACKER
====================================================
1. Add   2. View   3. Complete   4. Delete   5. Exit
Choice:
```

| Option | Action |
|---|---|
| `1` | Add a task |
| `2` | View all tasks |
| `3` | Mark a selected task as done |
| `4` | Delete a selected task |
| `5` | Exit the program |

## Example

```text
TASK TRACKER
1. Add   2. View   3. Complete   4. Delete   5. Exit
Choice: 1
Task Title: Read Python notes
Task added.

Choice: 2
1. Read Python notes [⏳ Pending]

Choice: 3
1. Read Python notes [⏳ Pending]
Task Number: 1
Task Completed.

Choice: 2
1. Read Python notes [✅ Done]
```

## How Data Is Stored

Each task is represented as a dictionary with two fields:

- `title`: The task title as a string.
- `done`: A Boolean value (`false` for pending tasks and `true` for completed tasks).

Example `tasks.json`:

```json
[
    {
        "title": "Read Python notes",
        "done": true
    },
    {
        "title": "Practise loops",
        "done": false
    }
]
```

The program uses `json.dump()` to save tasks and `json.load()` to read them. If the JSON file does not exist, the program starts with an empty task list. If the file contains malformed JSON or unexpected task data, it displays a message and starts with an empty list.

## Main Functions

| Function | Purpose |
|---|---|
| `load_tasks()` | Loads and validates saved tasks from `tasks.json`. |
| `save_tasks(tasks)` | Writes the task list to the JSON file. |
| `add_task(tasks)` | Gets a title, validates it, adds a pending task, and saves the list. |
| `view_task(tasks)` | Displays tasks and their current statuses. |
| `select_task(tasks)` | Validates a task number and returns the selected task's index. |
| `complete_task(tasks)` | Marks the selected task as done. |
| `delete_task(tasks)` | Removes the selected task and saves the updated list. |
| `main()` | Loads tasks and runs the menu loop until the user exits. |

## Limitations

- Tasks are stored locally in a JSON file; they are not synced between devices.
- Completing a task updates it in memory, but the current `complete_task()` function does **not** call `save_tasks(tasks)`. To preserve completed status after restarting, add `save_tasks(tasks)` after setting `tasks[index]["done"] = True`.
- The task list starts empty if the JSON file is missing or invalid.

## Future Improvements

- Save completed-task changes automatically.
- Add task due dates or priorities.
- Allow users to edit task titles.
- Add automated tests.

## Learning Concepts

This project practises Python functions, lists, dictionaries, loops, conditional statements, exception handling, file paths, JSON serialization, and input validation.

--

## is this repo insteresting?
YES! then do star 🌟 repo