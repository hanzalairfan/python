import tkinter as tk
from tkinter import messagebox, ttk
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv
import os

class BudgetEntry:
    def __init__(self, category, amount):
        self.category = category
        self.amount = amount

class MonthlyBudget:
    def __init__(self, month_name, income):
        self.month_name = month_name
        self.income = income
        self.expenses = []

    def add_expense(self, category, amount):
        self.expenses.append(BudgetEntry(category, amount))

    def get_total_expenses(self):
        return sum(entry.amount for entry in self.expenses)

    def get_savings(self):
        return self.income - self.get_total_expenses()

class BudgetTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Personal Budget Tracker")
        self.center_window(720, 560)

        self.all_months = []
        self.current_month = None

        label_font = ("Arial", 11)
        entry_font = ("Arial", 11)

        input_frame = tk.LabelFrame(root, text="Budget Input", padx=10, pady=10)
        input_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(input_frame, text="Month:", font=label_font).grid(row=0, column=0, sticky="e", padx=5, pady=5)
        months_list = ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]
        self.month_combobox = ttk.Combobox(input_frame, values=months_list, state="readonly", font=entry_font)
        self.month_combobox.grid(row=0, column=1, padx=5, pady=5)
        self.month_combobox.current(0)

        tk.Label(input_frame, text="Income:", font=label_font).grid(row=0, column=2, sticky="e", padx=5, pady=5)
        self.income_entry = tk.Entry(input_frame, font=entry_font)
        self.income_entry.grid(row=0, column=3, padx=5, pady=5)

        tk.Label(input_frame, text="Category:", font=label_font).grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.category_entry = tk.Entry(input_frame, font=entry_font)
        self.category_entry.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(input_frame, text="Amount:", font=label_font).grid(row=1, column=2, sticky="e", padx=5, pady=5)
        self.amount_entry = tk.Entry(input_frame, font=entry_font)
        self.amount_entry.grid(row=1, column=3, padx=5, pady=5)

        action_frame = tk.Frame(root)
        action_frame.pack(pady=5)

        tk.Button(action_frame, text="Add Expense", command=self.add_expense, width=15).grid(row=0, column=0, padx=5)
        tk.Button(action_frame, text="Finish Month", command=self.finish_month, width=15).grid(row=0, column=1, padx=5)
        tk.Button(action_frame, text="Show Summary", command=self.show_summary, width=15).grid(row=0, column=2, padx=5)
        tk.Button(action_frame, text="Show Visual Summary", command=self.show_visual_summary, width=18).grid(row=0, column=3, padx=5)

        self.tree = ttk.Treeview(root, columns=("Category", "Amount"), show='headings', height=10)
        self.tree.heading("Category", text="Category")
        self.tree.heading("Amount", text="Amount")
        self.tree.pack(fill="both", padx=10, pady=10, expand=True)

        scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.place(relx=1.0, rely=0.34, relheight=0.41, anchor='ne')

    def center_window(self, width, height):
        self.root.update_idletasks()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def add_expense(self):
        category = self.category_entry.get()
        amount_text = self.amount_entry.get()

        if not category or not amount_text:
            messagebox.showwarning("Input Error", "Please fill in category and amount.")
            return

        try:
            amount = float(amount_text)
            if self.current_month is None:
                month = self.month_combobox.get()
                income_text = self.income_entry.get()
                if not income_text:
                    messagebox.showwarning("Input Error", "Please enter income before adding expenses.")
                    return
                income = float(income_text)
                self.current_month = MonthlyBudget(month, income)

            self.current_month.add_expense(category, amount)
            self.tree.insert("", "end", values=(category, f"{amount:.2f}"))
            self.category_entry.delete(0, tk.END)
            self.amount_entry.delete(0, tk.END)

            if self.current_month.get_savings() < 0:
                messagebox.showwarning("Budget Warning", "Warning: Your expenses exceed your income!")
        except ValueError:
            messagebox.showerror("Invalid Input", "Amount and Income must be numbers.")

    def finish_month(self):
        if self.current_month is None:
            messagebox.showerror("Error", "Add income and at least one expense.")
            return

        self.all_months.append(self.current_month)

        filename = "budget_data.csv"
        file_exists = os.path.isfile(filename)

        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Month", "Income", "Category", "Amount"])
            for entry in self.current_month.expenses:
                writer.writerow([self.current_month.month_name, self.current_month.income, entry.category, entry.amount])

        messagebox.showinfo("Saved", f"Month '{self.current_month.month_name}' saved to CSV.")
        self.current_month = None
        self.month_combobox.current(0)
        self.income_entry.delete(0, tk.END)
        for i in self.tree.get_children():
            self.tree.delete(i)

    def show_summary(self):
        if not self.all_months:
            messagebox.showinfo("No Data", "No months saved yet.")
            return

        total_income = total_expense = total_savings = 0
        summary = ""

        for month in self.all_months:
            expenses = month.get_total_expenses()
            savings = month.get_savings()
            summary += f"Month: {month.month_name} | Income: {month.income} | Expenses: {expenses:.2f} | Savings: {savings:.2f}\n"
            total_income += month.income
            total_expense += expenses
            total_savings += savings

        summary += f"\nTotal Income: {total_income}"
        summary += f"\nTotal Expenses: {total_expense:.2f}"
        summary += f"\nTotal Savings: {total_savings:.2f}"

        summary_window = tk.Toplevel(self.root)
        summary_window.title("Budget Summary")
        text_area = tk.Text(summary_window, width=80, height=20)
        text_area.pack()
        text_area.insert(tk.END, summary)
        text_area.config(state='disabled')

    def show_visual_summary(self):
        if not self.all_months:
            messagebox.showinfo("No Data", "No months saved yet.")
            return

        visual_window = tk.Toplevel(self.root)
        visual_window.title("Select Month for Visualization")
        visual_window.geometry("300x120")

        tk.Label(visual_window, text="Choose a month:").pack(pady=10)
        month_names = [month.month_name for month in self.all_months]
        self.selected_month_var = tk.StringVar()
        month_combo = ttk.Combobox(visual_window, values=month_names, textvariable=self.selected_month_var, state="readonly")
        month_combo.pack(pady=5)
        month_combo.current(0)

        tk.Button(visual_window, text="Show Graph", command=self.plot_selected_month).pack(pady=5)

    def plot_selected_month(self):
        selected_name = self.selected_month_var.get()
        selected = next((m for m in self.all_months if m.month_name == selected_name), None)
        if not selected:
            messagebox.showerror("Error", "Selected month data not found.")
            return

        income = selected.income
        expenses = selected.get_total_expenses()
        savings = selected.get_savings()

        df = pd.DataFrame({
            'Category': ['Income', 'Expenses', 'Savings'],
            'Amount': [income, expenses, savings]
        })

        sns.set(style="whitegrid")
        plt.figure(figsize=(6, 4))
        sns.barplot(x='Category', y='Amount', data=df, palette="pastel")
        plt.title(f"Budget Overview - {selected.month_name}")
        plt.ylabel("Amount")
        plt.tight_layout()
        plt.show()


# Launch the app
root = tk.Tk()
app = BudgetTrackerApp(root)
root.mainloop()