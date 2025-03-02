import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import datetime
import csv
import json
import calendar
from datetime import timedelta
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from ics import Calendar, Event
from pathlib import Path
import base64
from io import BytesIO, StringIO
from dateutil import parser as date_parser
import pytz

# Set page configuration
st.set_page_config(
    page_title="Syllabus Analyzer & Scheduler",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the SyllabusAnalyzer class
class SyllabusAnalyzer:
    def __init__(self, reading_rate=15, writing_rate=15):
        """
        Initialize the SyllabusAnalyzer with customizable time metrics.
        
        Args:
            reading_rate: Minutes it takes to read 10 pages
            writing_rate: Minutes it takes to write 100 words
        """
        self.reading_rate = reading_rate
        self.writing_rate = writing_rate
        
        # Initialize storage for extracted assignments
        self.courses = {}
        self.all_assignments = []
        
        # Default available time slots (can be customized by the user)
        self.available_times = {
            'Monday': [{'start': '18:00', 'end': '21:00'}],
            'Tuesday': [{'start': '18:00', 'end': '21:00'}],
            'Wednesday': [{'start': '18:00', 'end': '21:00'}],
            'Thursday': [{'start': '18:00', 'end': '21:00'}],
            'Friday': [{'start': '16:00', 'end': '20:00'}],
            'Saturday': [{'start': '10:00', 'end': '18:00'}],
            'Sunday': [{'start': '10:00', 'end': '18:00'}]
        }
        
        # Schedule output
        self.schedule = []

    def add_syllabus(self, text, course_name):
        """
        Analyze a syllabus text and extract assignments.
        
        Args:
            text: The syllabus text content
            course_name: Name of the course this syllabus belongs to
        """
        st.info(f"Analyzing syllabus for {course_name}...")
        
        # Create course entry if doesn't exist
        if course_name not in self.courses:
            self.courses[course_name] = {
                'name': course_name,
                'assignments': []
            }
        
        # Split into paragraphs for better context
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        # Extract assignments using pattern matching techniques
        for i, paragraph in enumerate(paragraphs):
            # Look for assignment indicators
            if any(keyword in paragraph.lower() for keyword in 
                  ['assignment', 'essay', 'paper', 'project', 'homework', 
                   'reading', 'due', 'submit', 'deadline', 'exam', 'quiz']):
                
                # Extract assignment details
                assignment = self._extract_assignment_details(paragraph, i, course_name)
                
                if assignment:
                    # Add to course assignments
                    self.courses[course_name]['assignments'].append(assignment)
                    # Add to all assignments with course info
                    assignment_with_course = assignment.copy()
                    assignment_with_course['course'] = course_name
                    self.all_assignments.append(assignment_with_course)
        
        st.success(f"Found {len(self.courses[course_name]['assignments'])} assignments for {course_name}")
        return self.courses[course_name]['assignments']

    def _extract_assignment_details(self, text, paragraph_id, course_name):
        """
        Extract assignment details from a paragraph.
        
        Args:
            text: Original text
            paragraph_id: ID of the paragraph in the document
            course_name: Name of the course
            
        Returns:
            Dictionary containing assignment details
        """
        # Initialize assignment data
        assignment = {
            'id': f"{course_name}-{paragraph_id}",
            'name': None,
            'type': None,
            'due_date': None,
            'due_date_obj': None,
            'pages': 0,
            'word_count': 0,
            'estimated_time': 0,
            'raw_text': text
        }
        
        # Assignment name extraction - look for patterns like "Assignment 1:" or "Final Project:"
        name_match = re.search(r'([A-Za-z]+\s*\d*\s*:|\b[A-Za-z]+ (Assignment|Project|Paper|Essay|Exam|Quiz)[:\s])', text)
        if name_match:
            assignment['name'] = name_match.group(0).strip()
        else:
            # Default name based on content
            words = text.split()
            assignment['name'] = ' '.join(words[:min(5, len(words))]) + "..."
        
        # Determine assignment type
        if 'read' in text.lower() or 'pages' in text.lower():
            assignment['type'] = 'reading'
        elif any(word in text.lower() for word in ['write', 'essay', 'paper', 'report']):
            assignment['type'] = 'writing'
        elif 'project' in text.lower():
            assignment['type'] = 'project'
        elif 'exam' in text.lower() or 'quiz' in text.lower() or 'test' in text.lower():
            assignment['type'] = 'exam'
        else:
            assignment['type'] = 'assignment'
        
        # Extract due date using regex patterns
        date_pattern = re.compile(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2}(st|nd|rd|th)?(\s*,\s*\d{4})?', re.IGNORECASE)
        alt_date_pattern = re.compile(r'\b\d{1,2}[\/\-]\d{1,2}([\/\-]\d{2,4})?')
        
        # Find all date matches
        date_matches = date_pattern.findall(text)
        alt_date_matches = alt_date_pattern.findall(text)
        
        # Extract dates
        due_date_candidates = []
        if date_matches:
            for match in date_matches:
                if isinstance(match, tuple):
                    due_date_candidates.append(match[0])
                else:
                    due_date_candidates.append(match)
        
        if alt_date_matches:
            due_date_candidates.extend(alt_date_matches)
        
        # Look for due date indicators
        due_indicators = ['due', 'deadline', 'submit by', 'turn in by', 'due date']
        for indicator in due_indicators:
            if indicator in text.lower():
                indicator_pos = text.lower().find(indicator)
                closest_date = None
                min_distance = float('inf')
                
                for date_str in due_date_candidates:
                    if isinstance(date_str, str):
                        date_pos = text.lower().find(date_str.lower())
                        if date_pos > indicator_pos:  # Date appears after the indicator
                            distance = date_pos - indicator_pos
                            if distance < min_distance:
                                min_distance = distance
                                closest_date = date_str
                
                if closest_date:
                    assignment['due_date'] = closest_date
                    break
        
        # If no due date with indicator was found, use the first date
        if not assignment['due_date'] and due_date_candidates:
            assignment['due_date'] = due_date_candidates[0]
        
        # Try to parse the due date string to a datetime object
        if assignment['due_date']:
            try:
                # Add current year if not specified
                if not re.search(r'\d{4}', str(assignment['due_date'])):
                    current_year = datetime.datetime.now().year
                    assignment['due_date'] = f"{assignment['due_date']}, {current_year}"
                
                parsed_date = date_parser.parse(str(assignment['due_date']), fuzzy=True)
                assignment['due_date_obj'] = parsed_date
                assignment['due_date'] = parsed_date.strftime("%Y-%m-%d")
            except:
                # Keep as string if parsing fails
                pass
        
        # Extract page count for readings - look for different page patterns
        page_patterns = [
            r'pages\s+(\d+)-(\d+)',                # pages 1-10
            r'pages\s+(\d+)\s*-\s*(\d+)',          # pages 1 - 10
            r'pages\s+(\d+)',                      # pages 10
            r'(\d+)\s*-\s*(\d+)\s*pages',          # 1-10 pages
            r'(\d+)\s*pages',                      # 10 pages
            r'(\d+)\s*pgs',                        # 10 pgs
            r'(\d+)\s*pg',                         # 10 pg
            r'chapter\s+\d+,\s*pages\s+(\d+)-(\d+)',  # chapter 1, pages 1-10
            r'chapter\s+\d+,\s*pages\s+(\d+)'      # chapter 1, pages 10
        ]
        
        for pattern in page_patterns:
            page_match = re.search(pattern, text, re.IGNORECASE)
            if page_match:
                groups = page_match.groups()
                if len(groups) == 2:  # Range of pages
                    start_page = int(groups[0])
                    end_page = int(groups[1])
                    assignment['pages'] = end_page - start_page + 1
                elif len(groups) == 1:  # Single page count
                    assignment['pages'] = int(groups[0])
                break
        
        # Extract word count for written assignments
        word_match = re.search(r'(\d+)\s*(?:words|word)', text, re.IGNORECASE)
        if word_match:
            assignment['word_count'] = int(word_match.group(1))
        
        # Calculate estimated time based on assignment type and metrics
        if assignment['pages'] > 0:
            assignment['estimated_time'] += (assignment['pages'] / 10) * self.reading_rate
        
        if assignment['word_count'] > 0:
            assignment['estimated_time'] += (assignment['word_count'] / 100) * self.writing_rate
        
        # For assignments without specific metrics, assign default times based on type
        if assignment['estimated_time'] == 0:
            if assignment['type'] == 'reading':
                # Only set default time if we couldn't find pages
                if assignment['pages'] == 0:
                    # Look for words suggesting reading but no specific page count
                    if any(word in text.lower() for word in ['read', 'reading', 'textbook', 'chapter']):
                        # Assign estimated pages based on context
                        if 'chapter' in text.lower():
                            # Assume ~20 pages per chapter as default
                            implied_pages = 20
                            assignment['pages'] = implied_pages
                            assignment['estimated_time'] = (implied_pages / 10) * self.reading_rate
                        else:
                            assignment['estimated_time'] = 60  # Default 1 hour for unspecified readings
            elif assignment['type'] == 'writing':
                assignment['estimated_time'] = 120  # Default 2 hours for unspecified writing
            elif assignment['type'] == 'project':
                assignment['estimated_time'] = 180  # Default 3 hours for projects
            elif assignment['type'] == 'exam':
                assignment['estimated_time'] = 120  # Default 2 hours for exams
            else:
                assignment['estimated_time'] = 60  # Default 1 hour for other assignments
        
        return assignment

    def set_available_times(self, available_times):
        """
        Set custom available time slots.
        
        Args:
            available_times: Dictionary with days as keys and lists of time slots as values
        """
        self.available_times = available_times

    def generate_schedule(self, start_date=None, end_date=None):
        """
        Generate an optimized schedule based on assignments and available time slots.
        
        Args:
            start_date: Optional start date for the schedule (defaults to today)
            end_date: Optional end date for the schedule
            
        Returns:
            List of scheduled tasks
        """
        st.info("Generating schedule...")
        
        if not self.all_assignments:
            st.warning("No assignments to schedule.")
            return []
        
        # Default start date to today if not provided
        if start_date is None:
            start_date = datetime.datetime.now().date()
        elif isinstance(start_date, str):
            try:
                start_date = date_parser.parse(start_date).date()
            except:
                start_date = datetime.datetime.now().date()
        
        # Set a default end date if not provided (3 months from start)
        if end_date is None:
            end_date = start_date + datetime.timedelta(days=90)
        elif isinstance(end_date, str):
            try:
                end_date = date_parser.parse(end_date).date()
            except:
                end_date = start_date + datetime.timedelta(days=90)
        
        # Sort assignments by due date
        sorted_assignments = sorted(
            self.all_assignments, 
            key=lambda x: x.get('due_date_obj', datetime.datetime.max) if x.get('due_date_obj') else datetime.datetime.max
        )
        
        # Initialize schedule
        schedule = []
        
        # Map of weekday numbers to names
        weekday_names = {
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
            4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        }
        
        # Track scheduled assignments
        scheduled_assignments = set()
        
        # Keep track of days with scheduled tasks to spread them out
        scheduled_days = set()
        
        # Generate dates from start to end
        current_date = start_date
        
        # Try to space out assignments to have ~3 days between each task when possible
        min_days_between_tasks = 2  # Target at least 2 days between tasks
        
        while current_date <= end_date:
            # If we've had tasks in the past few days, skip this day to spread things out
            # unless we're getting close to a deadline
            if current_date.strftime("%Y-%m-%d") in scheduled_days:
                current_date += datetime.timedelta(days=1)
                continue
                
            # Check if this day is too close to previous scheduled days
            too_close = False
            for i in range(1, min_days_between_tasks + 1):
                check_date = current_date - datetime.timedelta(days=i)
                if check_date.strftime("%Y-%m-%d") in scheduled_days:
                    too_close = True
                    break
            
            # If this day is too close to previous tasks and we're not running out of time, 
            # skip to spread things out
            urgent_assignments = []
            for assignment in sorted_assignments:
                if assignment['id'] in scheduled_assignments:
                    continue
                
                # If assignment due within 7 days, consider it urgent
                if assignment.get('due_date_obj') and (assignment['due_date_obj'].date() - current_date).days <= 7:
                    urgent_assignments.append(assignment)
            
            # Skip this day if it's too close to previous tasks and there are no urgent assignments
            if too_close and not urgent_assignments:
                current_date += datetime.timedelta(days=1)
                continue
            
            weekday = current_date.weekday()
            day_name = weekday_names[weekday]
            
            # Tasks scheduled for this day
            day_has_tasks = False
            
            # Check if there are time slots available for this day
            if day_name in self.available_times:
                for time_slot in self.available_times[day_name]:
                    # Parse time slots
                    start_time = datetime.datetime.strptime(time_slot['start'], "%H:%M").time()
                    end_time = datetime.datetime.strptime(time_slot['end'], "%H:%M").time()
                    
                    # Calculate available minutes
                    start_minutes = start_time.hour * 60 + start_time.minute
                    end_minutes = end_time.hour * 60 + end_time.minute
                    available_minutes = end_minutes - start_minutes
                    
                    # Skip short time slots (less than 1 hour) to avoid cramming
                    if available_minutes < 60:
                        continue
                    
                    # Find assignments that fit in this time slot
                    remaining_minutes = available_minutes
                    start_minute = start_minutes
                    
                    # Prioritize assignments close to their due dates
                    for assignment in sorted_assignments:
                        if assignment['id'] in scheduled_assignments:
                            continue
                        
                        # Skip assignments already past due
                        if assignment.get('due_date_obj') and assignment['due_date_obj'].date() < current_date:
                            continue
                        
                        # Check if assignment fits in remaining time
                        est_time = assignment['estimated_time']
                        
                        # Allow breaking large assignments into smaller chunks
                        if est_time > remaining_minutes:
                            if remaining_minutes >= 45:  # Minimum 45 minutes per session
                                task_time = remaining_minutes
                                # Update the assignment's remaining time
                                assignment['estimated_time'] -= task_time
                            else:
                                continue
                        else:
                            task_time = est_time
                            scheduled_assignments.add(assignment['id'])
                        
                        # Calculate task time slot
                        task_start_time = datetime.time(int(start_minute // 60), int(start_minute % 60))
                        task_end_minute = start_minute + task_time
                        task_end_time = datetime.time(int(task_end_minute // 60), int(task_end_minute % 60))
                        
                        # Create schedule entry
                        schedule_entry = {
                            'date': current_date.strftime("%Y-%m-%d"),
                            'day': day_name,
                            'start_time': task_start_time.strftime("%H:%M"),
                            'end_time': task_end_time.strftime("%H:%M"),
                            'assignment_id': assignment['id'],
                            'assignment_name': assignment['name'],
                            'course': assignment['course'],
                            'duration_minutes': task_time,
                            'complete': False
                        }
                        
                        schedule.append(schedule_entry)
                        day_has_tasks = True
                        
                        # Update remaining time and start minute for next task
                        remaining_minutes -= task_time
                        start_minute += task_time
                        
                        # Only schedule one task per time slot to avoid cramming
                        break
            
            # Mark this day as having scheduled tasks
            if day_has_tasks:
                scheduled_days.add(current_date.strftime("%Y-%m-%d"))
            
            # Move to next day
            current_date += datetime.timedelta(days=1)
        
        # Check for unscheduled assignments
        unscheduled = [a['id'] for a in sorted_assignments if a['id'] not in scheduled_assignments]
        if unscheduled:
            st.warning(f"Could not schedule {len(unscheduled)} assignments due to time constraints.")
        
        self.schedule = schedule
        return schedule

    def export_to_ical(self):
        """
        Export schedule to iCalendar format for import into calendar applications.
        
        Returns:
            iCalendar string
        """
        if not self.schedule:
            st.warning("No schedule to export.")
            return None
        
        # Create a calendar
        cal = Calendar()
        
        # Add each scheduled task as an event
        for task in self.schedule:
            event = Event()
            
            # Create date objects
            task_date = date_parser.parse(task['date']).date()
            start_time = datetime.datetime.strptime(task['start_time'], "%H:%M").time()
            end_time = datetime.datetime.strptime(task['end_time'], "%H:%M").time()
            
            # Combine date and time
            start_datetime = datetime.datetime.combine(task_date, start_time)
            end_datetime = datetime.datetime.combine(task_date, end_time)
            
            # Set event properties
            event.name = f"{task['course']}: {task['assignment_name']}"
            event.begin = start_datetime
            event.end = end_datetime
            event.description = f"Work on {task['assignment_name']} for {task['course']}"
            event.location = 'Home/Library'
            
            # Add the event to the calendar
            cal.events.add(event)
        
        return cal.serialize()

    def export_to_csv(self):
        """
        Export schedule to CSV format.
        
        Returns:
            CSV string
        """
        if not self.schedule:
            st.warning("No schedule to export.")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(self.schedule)
        
        # Convert to CSV
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()
        
        return csv_string
    
    def get_workload_summary(self):
        """
        Generate a summary of the workload by course.
        
        Returns:
            Dictionary with course statistics
        """
        if not self.all_assignments:
            return {}
        
        # Group by course
        course_stats = {}
        for assignment in self.all_assignments:
            course = assignment['course']
            if course not in course_stats:
                course_stats[course] = {
                    'total_assignments': 0,
                    'total_time_minutes': 0,
                    'reading_pages': 0,
                    'writing_words': 0,
                    'assignments_by_type': {'reading': 0, 'writing': 0, 'project': 0, 'assignment': 0, 'exam': 0}
                }
            
            course_stats[course]['total_assignments'] += 1
            course_stats[course]['total_time_minutes'] += assignment['estimated_time']
            course_stats[course]['reading_pages'] += assignment['pages']
            course_stats[course]['writing_words'] += assignment['word_count']
            
            # Update assignment type counter
            if assignment['type'] in course_stats[course]['assignments_by_type']:
                course_stats[course]['assignments_by_type'][assignment['type']] += 1
            else:
                course_stats[course]['assignments_by_type']['assignment'] += 1
        
        # Add total hours
        for course in course_stats:
            course_stats[course]['total_hours'] = round(course_stats[course]['total_time_minutes'] / 60, 1)
        
        return course_stats

    def clear_data(self):
        """Clear all data"""
        self.courses = {}
        self.all_assignments = []
        self.schedule = []

# Function to create a download link
def get_download_link(content, filename, text):
    """Generate a download link for a file"""
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Initialize session state variables if they don't exist
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = SyllabusAnalyzer()

if 'show_sample' not in st.session_state:
    st.session_state.show_sample = False

if 'sample_calendar_loaded' not in st.session_state:
    st.session_state.sample_calendar_loaded = False

# Sample syllabi for Spring 2025 semester with explicit page counts
sample_syllabi = {
    "CS101": """
CS 101: Introduction to Computer Science
Spring 2025

Course Schedule:

Week 1: Introduction to Programming
Reading: Chapter 1, pages 1-15 due February 7, 2025

Week 3: Control Structures
Reading: Chapter 3, pages 50-70, due February 18, 2025
Assignment 1: Decision tree implementation, due February 21, 2025

Week 5: Midterm Exam Preparation
Reading: Review chapters 1-4, pages 1-90, due March 5, 2025
Midterm Exam: March 7, 2025

Week 10: Data Structures
Reading: Chapter 6, pages 120-135, due April 11, 2025

Final Project: Create a simple game using Python, 800 words report, due May 9, 2025
""",
    "ENG201": """
ENG 201: Creative Writing
Spring 2025

Assignments:

Reading: Introduction to Creative Writing, pages 10-30, due February 10, 2025
Short Story: Write a 400-word short story due February 20, 2025
Reading: Modern Poetry Anthology, pages 45-65, due March 5, 2025
Research Paper Proposal: 200 words, due March 15, 2025
Reading Assignment: Modern authors anthology, pages 80-100, due March 30, 2025
Final Portfolio: Collection of your work, due May 5, 2025
""",
    "MATH150": """
MATH 150: Calculus I
Spring 2025

Course Assignments:

Reading: Introduction to Limits, pages 15-25, due February 5, 2025
Homework 1: Problems from Chapter 1, pages 25-35, due February 12, 2025
Reading: Differentiation Techniques, pages 40-55, due February 25, 2025
Midterm Exam: March 5, 2025
Reading: Integration Methods, pages 60-75, due April 5, 2025
Homework 2: Reading and problems, pages 70-85, due April 15, 2025
Final Project: Applied calculus problem, 300 words, due April 30, 2025
"""
}

# Sample calendar of commitments
sample_calendar = {
    'Monday': [
        {'start': '09:00', 'end': '10:15', 'name': 'CS101 Lecture'},
        {'start': '12:30', 'end': '13:45', 'name': 'MATH150 Lecture'},
        {'start': '15:00', 'end': '16:30', 'name': 'Basketball Practice'}
    ],
    'Tuesday': [
        {'start': '11:00', 'end': '12:15', 'name': 'ENG201 Workshop'},
        {'start': '14:00', 'end': '15:00', 'name': 'Academic Advising'},
        {'start': '19:00', 'end': '20:30', 'name': 'Study Group'}
    ],
    'Wednesday': [
        {'start': '09:00', 'end': '10:15', 'name': 'CS101 Lecture'},
        {'start': '12:30', 'end': '13:45', 'name': 'MATH150 Lecture'},
        {'start': '16:00', 'end': '18:00', 'name': 'Part-time Job'}
    ],
    'Thursday': [
        {'start': '11:00', 'end': '12:15', 'name': 'ENG201 Workshop'},
        {'start': '15:00', 'end': '16:30', 'name': 'CS101 Lab'},
        {'start': '19:00', 'end': '21:00', 'name': 'Social Club'}
    ],
    'Friday': [
        {'start': '10:00', 'end': '11:30', 'name': 'MATH150 Recitation'},
        {'start': '13:00', 'end': '17:00', 'name': 'Part-time Job'}
    ],
    'Saturday': [
        {'start': '10:00', 'end': '12:00', 'name': 'Volunteer Work'},
        {'start': '15:00', 'end': '17:00', 'name': 'Gym Session'}
    ],
    'Sunday': [
        {'start': '19:00', 'end': '20:30', 'name': 'Family Video Call'}
    ]
}

# Define days of the week (globally available)
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Create a modern sidebar
with st.sidebar:
    st.title("Syllabus Analyzer")
    
    # Add a sample data option
    if st.button("Load Sample Data"):
        st.session_state.show_sample = True
        
        # First clear any existing data
        st.session_state.analyzer.clear_data()
        
        # Load sample syllabi
        for course, syllabus in sample_syllabi.items():
            st.session_state.analyzer.add_syllabus(syllabus, course)
            
        # Load sample calendar
        if 'imported_events' not in st.session_state:
            st.session_state.imported_events = []
        else:
            st.session_state.imported_events = []
            
        # Add sample calendar events to session state
        for day, events in sample_calendar.items():
            for event in events:
                # Convert day name to a sample date for visualization
                day_num = days.index(day)
                # Find next occurrence of this day
                today = datetime.datetime.now().date()
                days_ahead = day_num - today.weekday()
                if days_ahead < 0:
                    days_ahead += 7
                next_day = today + datetime.timedelta(days=days_ahead)
                
                event_date = next_day.strftime("%Y-%m-%d")
                
                st.session_state.imported_events.append({
                    'date': event_date,
                    'day': day,
                    'start_time': event['start'],
                    'end_time': event['end'],
                    'name': event['name'],
                    'type': 'existing_event'
                })
                
        # Generate available times from sample calendar
        available_times = {}
        
        for day in days:
            day_busy_times = sample_calendar.get(day, [])
            
            # Sort busy times
            day_busy_times.sort(key=lambda x: x['start'])
            
            # Start with full day availability (8 AM to 10 PM)
            default_start = datetime.time(8, 0)
            default_end = datetime.time(22, 0)
            day_available_times = [{'start': default_start.strftime("%H:%M"), 'end': default_end.strftime("%H:%M")}]
            
            # Subtract busy times
            for busy in day_busy_times:
                busy_start = datetime.datetime.strptime(busy['start'], "%H:%M").time()
                busy_end = datetime.datetime.strptime(busy['end'], "%H:%M").time()
                
                new_available_times = []
                
                for available in day_available_times:
                    avail_start = datetime.datetime.strptime(available['start'], "%H:%M").time()
                    avail_end = datetime.datetime.strptime(available['end'], "%H:%M").time()
                    
                    # Case 1: Busy time completely outside available time
                    if busy_end <= avail_start or busy_start >= avail_end:
                        new_available_times.append(available)
                        continue
                    
                    # Case 2: Busy time at the beginning of available time
                    if busy_start <= avail_start and busy_end < avail_end:
                        new_available_times.append({'start': busy_end.strftime("%H:%M"), 'end': avail_end.strftime("%H:%M")})
                        continue
                    
                    # Case 3: Busy time at the end of available time
                    if busy_start > avail_start and busy_end >= avail_end:
                        new_available_times.append({'start': avail_start.strftime("%H:%M"), 'end': busy_start.strftime("%H:%M")})
                        continue
                    
                    # Case 4: Busy time in the middle of available time
                    if busy_start > avail_start and busy_end < avail_end:
                        new_available_times.append({'start': avail_start.strftime("%H:%M"), 'end': busy_start.strftime("%H:%M")})
                        new_available_times.append({'start': busy_end.strftime("%H:%M"), 'end': avail_end.strftime("%H:%M")})
                        continue
                    
                    # Case 5: Busy time covers all available time
                    if busy_start <= avail_start and busy_end >= avail_end:
                        continue
                
                day_available_times = new_available_times
            
            # Filter out time slots that are too short (less than 30 minutes)
            day_available_times = [slot for slot in day_available_times if 
                                  (datetime.datetime.strptime(slot['end'], "%H:%M") - 
                                  datetime.datetime.strptime(slot['start'], "%H:%M")).seconds / 60 >= 30]
            
            if day_available_times:
                available_times[day] = day_available_times
        
        # Update analyzer with available times
        st.session_state.analyzer.set_available_times(available_times)
        st.session_state.sample_calendar_loaded = True
        
        st.success("Sample data loaded! Syllabi and calendar imported.")
    
    if st.button("Clear All Data"):
        st.session_state.analyzer.clear_data()
        st.session_state.show_sample = False
        if 'imported_events' in st.session_state:
            st.session_state.imported_events = []
        st.session_state.sample_calendar_loaded = False
        st.success("All data cleared!")
    
    st.header("Time Parameters")
    
    # Create time parameter inputs
    reading_rate = st.number_input(
        "Reading Rate (minutes per 10 pages)", 
        min_value=1, 
        value=st.session_state.analyzer.reading_rate
    )
    
    writing_rate = st.number_input(
        "Writing Rate (minutes per 100 words)", 
        min_value=1, 
        value=st.session_state.analyzer.writing_rate
    )
    
    # Update analyzer parameters
    st.session_state.analyzer.reading_rate = reading_rate
    st.session_state.analyzer.writing_rate = writing_rate
    
    # Available time slots configuration
    st.header("Available Time Slots")
    
    # Define days of the week
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Calendar import option
    with st.expander("Import Calendar (iCal/ICS)"):
        st.write("Upload your calendar file to automatically block off times when you have existing commitments.")
        calendar_file = st.file_uploader("Upload iCal/ICS file", type=['ics', 'ical', 'icalendar'], key="calendar_upload")
        
        if calendar_file is not None:
            try:
                from ics import Calendar as IcsCalendar
                
                # Read calendar file
                calendar_content = calendar_file.read().decode()
                imported_calendar = IcsCalendar(calendar_content)
                
                # Process events to identify busy times
                busy_times = {}
                events_found = 0
                
                for event in imported_calendar.events:
                    if event.begin and event.end:
                        event_start = event.begin.datetime
                        event_end = event.end.datetime
                        
                        # Make datetime timezone naive if it's timezone aware
                        if event_start.tzinfo is not None:
                            event_start = event_start.replace(tzinfo=None)
                        if event_end.tzinfo is not None:
                            event_end = event_end.replace(tzinfo=None)
                        
                        # Only consider events within a reasonable timeframe (next 3 months)
                        max_future_date = datetime.datetime.now() + datetime.timedelta(days=90)
                        if event_start > max_future_date:
                            continue
                        
                        events_found += 1
                        
                        # Get day of week
                        day_name = days[event_start.weekday()]
                        
                        # Format times
                        start_time_str = event_start.strftime("%H:%M")
                        end_time_str = event_end.strftime("%H:%M")
                        
                        # Add to busy times
                        if day_name not in busy_times:
                            busy_times[day_name] = []
                        
                        busy_times[day_name].append({
                            'start': start_time_str,
                            'end': end_time_str,
                            'name': event.name
                        })
                
                st.success(f"Successfully imported {events_found} events!")
                
                # Show busy times
                if busy_times:
                    st.subheader("Busy Times (will be blocked off)")
                    for day, times in busy_times.items():
                        with st.expander(f"{day} ({len(times)} events)"):
                            for time_slot in times:
                                st.write(f"{time_slot['start']} - {time_slot['end']}: {time_slot['name']}")
                    
                    # Button to generate available times
                    if st.button("Generate Available Time Slots"):
                        # Default available hours (8 AM to 10 PM)
                        default_start = datetime.time(8, 0)
                        default_end = datetime.time(22, 0)
                        
                        # Generate available times by subtracting busy times
                        available_times = {}
                        
                        for day in days:
                            day_busy_times = busy_times.get(day, [])
                            
                            # Sort busy times
                            day_busy_times.sort(key=lambda x: x['start'])
                            
                            # Start with full day availability
                            day_available_times = [{'start': default_start.strftime("%H:%M"), 'end': default_end.strftime("%H:%M")}]
                            
                            # Subtract busy times
                            for busy in day_busy_times:
                                busy_start = datetime.datetime.strptime(busy['start'], "%H:%M").time()
                                busy_end = datetime.datetime.strptime(busy['end'], "%H:%M").time()
                                
                                new_available_times = []
                                
                                for available in day_available_times:
                                    avail_start = datetime.datetime.strptime(available['start'], "%H:%M").time()
                                    avail_end = datetime.datetime.strptime(available['end'], "%H:%M").time()
                                    
                                    # Case 1: Busy time completely outside available time
                                    if busy_end <= avail_start or busy_start >= avail_end:
                                        new_available_times.append(available)
                                        continue
                                    
                                    # Case 2: Busy time at the beginning of available time
                                    if busy_start <= avail_start and busy_end < avail_end:
                                        new_available_times.append({'start': busy_end.strftime("%H:%M"), 'end': avail_end.strftime("%H:%M")})
                                        continue
                                    
                                    # Case 3: Busy time at the end of available time
                                    if busy_start > avail_start and busy_end >= avail_end:
                                        new_available_times.append({'start': avail_start.strftime("%H:%M"), 'end': busy_start.strftime("%H:%M")})
                                        continue
                                    
                                    # Case 4: Busy time in the middle of available time
                                    if busy_start > avail_start and busy_end < avail_end:
                                        new_available_times.append({'start': avail_start.strftime("%H:%M"), 'end': busy_start.strftime("%H:%M")})
                                        new_available_times.append({'start': busy_end.strftime("%H:%M"), 'end': avail_end.strftime("%H:%M")})
                                        continue
                                    
                                    # Case 5: Busy time covers all available time
                                    if busy_start <= avail_start and busy_end >= avail_end:
                                        continue
                                
                                day_available_times = new_available_times
                            
                            # Filter out time slots that are too short (less than 30 minutes)
                            day_available_times = [slot for slot in day_available_times if 
                                                  (datetime.datetime.strptime(slot['end'], "%H:%M") - 
                                                  datetime.datetime.strptime(slot['start'], "%H:%M")).seconds / 60 >= 30]
                            
                            if day_available_times:
                                available_times[day] = day_available_times
                        
                        # Update analyzer with available times
                        if available_times:
                            st.session_state.analyzer.set_available_times(available_times)
                            st.success("Available time slots generated based on your calendar!")
                        else:
                            st.warning("No available time slots could be generated. Please check your calendar.")
            
            except Exception as e:
                st.error(f"Error processing calendar file: {str(e)}")
                st.info("Make sure the file is in valid iCalendar format.")
    
    # Create a dictionary to store available times
    available_times = {}
    
    # Create time slot inputs for each day
    selected_day = st.selectbox("Select Day", days)
    
    # Time slots for the selected day
    col1, col2 = st.columns(2)
    with col1:
        start_time = st.time_input("Start Time", datetime.time(18, 0))
    with col2:
        end_time = st.time_input("End Time", datetime.time(21, 0))
    
    # Button to add the time slot
    if st.button("Set Time Slot"):
        # Convert time to string format
        start_str = start_time.strftime("%H:%M")
        end_str = end_time.strftime("%H:%M")
        
        # Update available times
        available_times[selected_day] = [{'start': start_str, 'end': end_str}]
        st.session_state.analyzer.set_available_times(available_times)
        st.success(f"Time slot for {selected_day} updated: {start_str} - {end_str}")

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["📝 Add Syllabi", "📊 Analysis", "📅 Schedule", "⚙️ Help"])

# Tab 1: Add Syllabi
with tab1:
    st.header("Add Syllabi")
    
    # Option to use sample syllabi
    if st.session_state.show_sample:
        st.info("Sample syllabi loaded! Select a course below to analyze.")
        for course, syllabus in sample_syllabi.items():
            if st.button(f"Analyze {course} Syllabus"):
                st.session_state.analyzer.add_syllabus(syllabus, course)
    
    # Form for adding a new syllabus
    st.subheader("Add New Syllabus")
    course_name = st.text_input("Course Name", key="new_course")
    syllabus_text = st.text_area("Paste Syllabus Text", height=300, key="new_syllabus")
    
    if st.button("Analyze Syllabus", key="analyze_pasted_btn"):
        if course_name and syllabus_text:
            st.session_state.analyzer.add_syllabus(syllabus_text, course_name)
        else:
            st.error("Please enter both course name and syllabus text")
    
    # Upload syllabus file - without using expander
    st.subheader("Upload Syllabus File")
    uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'doc', 'docx'], key="file_upload")
    course_name_upload = st.text_input("Course Name", key="upload_course_name")
    
    if uploaded_file is not None and course_name_upload:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        try:
            if file_extension == 'txt':
                # Process text files
                syllabus_content = uploaded_file.read().decode()
            
            elif file_extension == 'pdf':
                # Use PyPDF2 for PDF files
                import io
                import PyPDF2
                
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                syllabus_content = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    syllabus_content += page.extract_text() + "\n"
            
            elif file_extension in ['doc', 'docx']:
                # Use python-docx for Word documents
                import io
                import docx
                
                doc_reader = docx.Document(io.BytesIO(uploaded_file.read()))
                syllabus_content = ""
                for para in doc_reader.paragraphs:
                    syllabus_content += para.text + "\n"
            
            # Show a preview of the extracted text
            st.subheader("Preview Extracted Text")
            st.text_area("Content Preview", 
                         value=syllabus_content[:500] + "..." if len(syllabus_content) > 500 else syllabus_content,
                         height=150,
                         disabled=True)
            
            if st.button("Process Uploaded Syllabus", key="process_upload_btn"):
                st.session_state.analyzer.add_syllabus(syllabus_content, course_name_upload)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("If you're having trouble with document extraction, try copying the text and pasting it directly in the 'Add New Syllabus' section.")
    
    # Display added courses
    if st.session_state.analyzer.courses:
        st.header("Courses Added")
        
        # Create columns for the course list
        cols = st.columns(3)
        
        # Display courses in columns
        for i, (course, data) in enumerate(st.session_state.analyzer.courses.items()):
            col_idx = i % 3
            with cols[col_idx]:
                # Create a card-like display for each course
                st.markdown(f"""
                <div style="
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 10px;
                    margin-bottom: 10px;
                    background-color: #f9f9f9;
                ">
                    <h3>{course}</h3>
                    <p>Assignments: {len(data['assignments'])}</p>
                </div>
                """, unsafe_allow_html=True)

# Tab 2: Analysis
with tab2:
    st.header("Course Analysis")
    
    if not st.session_state.analyzer.all_assignments:
        st.info("No courses added yet. Please add syllabi in the 'Add Syllabi' tab or click 'Load Sample Data' in the sidebar.")
        
        if not st.session_state.show_sample:
            st.markdown("""
            ### 📊 Want to see a complete demo?
            
            Click the **Load Sample Data** button in the sidebar to:
            
            1. Load three sample syllabi for Spring 2025 (CS101, ENG201, MATH150)
            2. Import a sample calendar with classes, study groups, work, and activities
            3. Generate available study time slots around your commitments
            
            This will show you how the system works with real-world data!
            """)
    else:
        # Workload summary
        workload = st.session_state.analyzer.get_workload_summary()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Prepare data for charts
            courses = list(workload.keys())
            hours = [stats['total_hours'] for course, stats in workload.items()]
            
            # Create bar chart of estimated hours
            fig = px.bar(
                x=courses, 
                y=hours,
                labels={'x': 'Course', 'y': 'Estimated Hours'},
                title='Workload Distribution by Course',
                color=hours,
                color_continuous_scale='Viridis'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create pie chart of assignment types
            assignment_types = {'reading': 0, 'writing': 0, 'project': 0, 'assignment': 0, 'exam': 0}
            
            for course, stats in workload.items():
                for atype, count in stats['assignments_by_type'].items():
                    assignment_types[atype] += count
            
            # Filter out zero values
            assignment_types = {k: v for k, v in assignment_types.items() if v > 0}
            
            if assignment_types:
                fig2 = px.pie(
                    names=list(assignment_types.keys()),
                    values=list(assignment_types.values()),
                    title='Assignment Types Distribution',
                    color=list(assignment_types.keys()),
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                
                st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            st.subheader("Course Summary")
            
            for course, stats in workload.items():
                with st.expander(course):
                    st.write(f"**Total assignments:** {stats['total_assignments']}")
                    st.write(f"**Estimated work:** {stats['total_hours']} hours")
                    st.write(f"**Reading:** {stats['reading_pages']} pages")
                    st.write(f"**Writing:** {stats['writing_words']} words")
                    
                    # Assignment breakdown
                    st.write("**Assignment types:**")
                    for atype, count in stats['assignments_by_type'].items():
                        if count > 0:
                            st.write(f"- {atype.capitalize()}: {count}")
        
        # Timeline visualization
        st.subheader("Assignment Timeline")
        
        # Prepare data for timeline
        timeline_data = []
        
        for assignment in st.session_state.analyzer.all_assignments:
            if assignment.get('due_date'):
                timeline_data.append({
                    'Task': assignment['name'],
                    'Course': assignment['course'],
                    'Start': datetime.datetime.now().date(),
                    'Finish': assignment.get('due_date_obj', datetime.datetime.now()).date() if isinstance(assignment.get('due_date_obj'), datetime.datetime) else datetime.datetime.now().date(),
                    'Type': assignment['type']
                })
        
        if timeline_data:
            # Convert to DataFrame
            df_timeline = pd.DataFrame(timeline_data)
            
            # Create Gantt chart
            fig3 = px.timeline(
                df_timeline, 
                x_start="Start", 
                x_end="Finish", 
                y="Task",
                color="Course",
                hover_name="Task",
                opacity=0.8
            )
            
            # Update layout
            fig3.update_yaxes(autorange="reversed")
            fig3.update_layout(
                title="Assignment Timeline",
                xaxis_title="Date",
                yaxis_title="Assignment",
                legend_title="Course"
            )
            
            st.plotly_chart(fig3, use_container_width=True)
        
        # All assignments table
        st.subheader("All Assignments")
        
        # Convert assignments to DataFrame
        assignments_df = pd.DataFrame(st.session_state.analyzer.all_assignments)
        
        if not assignments_df.empty:
            # Select and rename columns for display
            display_columns = {
                'course': 'Course',
                'name': 'Assignment',
                'type': 'Type',
                'due_date': 'Due Date',
                'pages': 'Pages',
                'word_count': 'Words',
                'estimated_time': 'Est. Time (min)'
            }
            
            # Filter and rename columns
            display_df = assignments_df[[col for col in display_columns.keys() if col in assignments_df.columns]]
            display_df = display_df.rename(columns=display_columns)
            
            # Format the dataframe
            display_df = display_df.fillna('')
            
            # Display as interactive table
            st.dataframe(display_df, use_container_width=True)

# Tab 3: Schedule
with tab3:
    st.header("Schedule Generator")
    
    if not st.session_state.analyzer.all_assignments:
        st.info("No courses added yet. Please add syllabi in the 'Add Syllabi' tab or click 'Load Sample Data' in the sidebar.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date", 
                datetime.datetime.now().date(),
                key="schedule_start_date"
            )
        
        with col2:
            end_date = st.date_input(
                "End Date (Optional)", 
                datetime.datetime.now().date() + datetime.timedelta(days=90),
                key="schedule_end_date"
            )
        
        if st.button("Generate Schedule", key="generate_schedule_btn"):
            schedule = st.session_state.analyzer.generate_schedule(
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            st.success(f"Schedule generated with {len(schedule)} study tasks!")
            if st.session_state.sample_calendar_loaded:
                st.info("Your study schedule has been created around your class times and other commitments! Check out the Calendar View below to see how it fits into your week.")
        
        # Sample data CTA if no schedule yet
        if not st.session_state.analyzer.schedule and not st.session_state.show_sample:
            st.info("📊 Tip: Click 'Load Sample Data' in the sidebar to see a complete example with course syllabi and a weekly schedule of classes and activities.")

        
        # Display schedule if available
        if st.session_state.analyzer.schedule:
            st.subheader("Generated Schedule")
            
            # Convert to DataFrame for display
            schedule_df = pd.DataFrame(st.session_state.analyzer.schedule)
            
            # Select and rename columns
            display_columns = {
                'date': 'Date',
                'day': 'Day',
                'start_time': 'Start',
                'end_time': 'End',
                'course': 'Course',
                'assignment_name': 'Assignment',
                'duration_minutes': 'Duration (min)'
            }
            
            # Format the dataframe
            display_df = schedule_df[[col for col in display_columns.keys() if col in schedule_df.columns]]
            display_df = display_df.rename(columns=display_columns)
            
            # Display as interactive table
            st.dataframe(display_df, use_container_width=True)
            
            # Calendar view
            st.subheader("Calendar View")
            
            # Prepare calendar data
            calendar_data = []
            for task in st.session_state.analyzer.schedule:
                task_date = date_parser.parse(task['date']).date()
                calendar_data.append({
                    'Date': task_date,
                    'Task': f"{task['course']}: {task['assignment_name']}",
                    'Time': f"{task['start_time']} - {task['end_time']}",
                    'Duration': task['duration_minutes']
                })
            
            # Group by date
            calendar_df = pd.DataFrame(calendar_data)
            if not calendar_df.empty:
                # Create date range for calendar
                min_date = calendar_df['Date'].min()
                max_date = calendar_df['Date'].max()
                
                # Create month view
                selected_month = st.date_input(
                    "Select Month View", 
                    min_date,
                    key="calendar_month"
                )
                
                month_start = datetime.date(selected_month.year, selected_month.month, 1)
                if selected_month.month == 12:
                    month_end = datetime.date(selected_month.year + 1, 1, 1) - datetime.timedelta(days=1)
                else:
                    month_end = datetime.date(selected_month.year, selected_month.month + 1, 1) - datetime.timedelta(days=1)
                
                # Create calendar grid
                month_name = selected_month.strftime("%B %Y")
                st.write(f"### {month_name}")
                
                # Create a calendar grid with 7 columns (weekdays)
                cal = calendar.monthcalendar(selected_month.year, selected_month.month)
                
                # Create a HTML table for the calendar
                html_cal = f"<table style='width:100%; border-collapse: collapse;'>"
                html_cal += f"<tr><th style='width:14%; text-align:center; padding:8px; background-color:#f2f2f2;'>Mon</th><th style='width:14%; text-align:center; padding:8px; background-color:#f2f2f2;'>Tue</th><th style='width:14%; text-align:center; padding:8px; background-color:#f2f2f2;'>Wed</th><th style='width:14%; text-align:center; padding:8px; background-color:#f2f2f2;'>Thu</th><th style='width:14%; text-align:center; padding:8px; background-color:#f2f2f2;'>Fri</th><th style='width:14%; text-align:center; padding:8px; background-color:#f2f2f2;'>Sat</th><th style='width:14%; text-align:center; padding:8px; background-color:#f2f2f2;'>Sun</th></tr>"
                
                for week in cal:
                    html_cal += "<tr>"
                    for day in week:
                        if day == 0:
                            # Empty cell
                            html_cal += "<td style='border:1px solid #ddd; padding:8px; vertical-align:top; height:100px;'></td>"
                        else:
                            # Create the date
                            date_val = datetime.date(selected_month.year, selected_month.month, day)
                            
                            # Check if there are tasks or events on this day
                            day_tasks = calendar_df[calendar_df['Date'] == date_val]
                            
                            day_events = []
                            
                            # Add scheduled study tasks for this day
                            for _, task in day_tasks.iterrows():
                                day_events.append({
                                    'start_time': task['Time'].split(' - ')[0],
                                    'end_time': task['Time'].split(' - ')[1],
                                    'name': task['Task'],
                                    'type': 'study_task'
                                })
                            
                            # Check for imported events on this day
                            if 'imported_events' in st.session_state:
                                # Get day name for this date
                                day_name = date_val.strftime('%A')
                                
                                # Find events for this day of week that recur weekly
                                for event in st.session_state.imported_events:
                                    if event['day'] == day_name:
                                        day_events.append({
                                            'start_time': event['start_time'],
                                            'end_time': event['end_time'],
                                            'name': event['name'],
                                            'type': 'existing_event'
                                        })
                            
                            # Sort all events chronologically by start time
                            day_events.sort(key=lambda x: x['start_time'])
                            
                            # Style based on whether there are tasks or events
                            has_study_tasks = any(e['type'] == 'study_task' for e in day_events)
                            has_existing_events = any(e['type'] == 'existing_event' for e in day_events)
                            
                            if has_study_tasks and has_existing_events:
                                bg_color = "#fff9c4"  # Yellow for days with both types
                            elif has_study_tasks:
                                bg_color = "#e6f7ff"  # Light blue for days with study tasks
                            elif has_existing_events:
                                bg_color = "#f9f9f9"  # Light gray for days with events
                            else:
                                bg_color = "#ffffff"  # White for days without tasks or events
                            
                            # Highlight today
                            if date_val == datetime.datetime.now().date():
                                day_style = f"font-weight:bold; background-color:#ffeb3b;"
                            else:
                                day_style = f"background-color:{bg_color};"
                            
                            html_cal += f"<td style='border:1px solid #ddd; padding:8px; vertical-align:top; height:100px; {day_style}'>"
                            html_cal += f"<div style='text-align:right;'>{day}</div>"
                            
                            # Add all events chronologically with appropriate styling
                            for event in day_events:
                                if event['type'] == 'existing_event':
                                    bg_color = "#ffebee"  # Light red for existing events
                                else:
                                    bg_color = "#e1f5fe"  # Light blue for study tasks
                                
                                html_cal += f"<div style='font-size:12px; background-color:{bg_color}; margin:2px; padding:4px; border-radius:4px;'>"
                                html_cal += f"{event['start_time']}-{event['end_time']}<br>{event['name']}</div>"
                            
                            html_cal += "</td>"
                    html_cal += "</tr>"
                
                html_cal += "</table>"
                
                # Display the calendar
                st.markdown(html_cal, unsafe_allow_html=True)
            
            # Export options
            st.subheader("Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Export to CSV"):
                    csv_content = st.session_state.analyzer.export_to_csv()
                    if csv_content:
                        st.download_button(
                            label="Download CSV",
                            data=csv_content,
                            file_name="schedule.csv",
                            mime="text/csv"
                        )
            
            with col2:
                if st.button("Export to Calendar (iCal)"):
                    ical_content = st.session_state.analyzer.export_to_ical()
                    if ical_content:
                        st.download_button(
                            label="Download iCal File",
                            data=ical_content,
                            file_name="schedule.ics",
                            mime="text/calendar"
                        )

# Tab 4: Help
with tab4:
    st.header("How to Use This Tool")
    
    st.markdown("""
    ### Syllabus Analyzer & Assignment Scheduler
    
    This tool helps you analyze course syllabi, extract assignments and deadlines, and create an optimized study schedule based on your available time.
    
    #### 🔍 See a Complete Example First
    
    The easiest way to understand how this tool works is to click the **Load Sample Data** button in the sidebar. This will:
    
    1. Load three sample syllabi for Spring 2025 (CS101, ENG201, MATH150)
    2. Import a sample weekly calendar with classes, work, and activities
    3. Generate available study time slots around your commitments
    
    Then go to the Schedule tab and click "Generate Schedule" to see how the system arranges your study time around your existing commitments.
    
    #### Step 1: Add Your Syllabi
    - Use the "Add Syllabi" tab to input your course information
    - You can paste syllabus text, upload files, or use sample data
    - The tool will extract assignments, due dates, and workload information
    
    #### Step 2: Review Analysis
    - Go to the "Analysis" tab to see your workload distribution
    - Check the assignment timeline to visualize deadlines
    - Verify all extracted assignments are correct
    
    #### Step 3: Set Up Your Schedule
    - Use the sidebar to customize your time parameters
    - Set available time slots for each day of the week
    - Import your existing calendar to automatically block off busy times
    - Go to the "Schedule" tab and generate your optimized schedule
    
    #### Step 4: Export Your Schedule
    - Export to CSV for spreadsheet applications
    - Export to iCalendar format for Google Calendar, Apple Calendar, etc.
    
    ### Tips for Best Results
    
    - Provide complete syllabus text with clear assignment descriptions
    - Include due dates in standard formats (e.g., "March 15, 2025" or "3/15/2025")
    - Specify page counts and word counts when available
    - Set realistic reading and writing rates based on your abilities
    - Configure available time slots that match your actual availability
    - Import your calendar to ensure no scheduling conflicts
    
    ### Sample Syllabi Format
    
    The tool works best when syllabi include clear assignment information:
    
    ```
    Assignment 1: Essay on [topic], 500 words, due March 15, 2025
    Reading: Chapter 3, pages 45-60, due February 22, 2025
    Final Project: Research paper, 1500 words, due May 10, 2025
    ```
    """)
    
    # Sample syllabus format
    st.subheader("View Sample Data")
    
    tab_syllabus, tab_calendar = st.tabs(["Sample Syllabus", "Sample Calendar"])
    
    with tab_syllabus:
        st.code(sample_syllabi["CS101"], language="text")
    
    with tab_calendar:
        # Create a dataframe of the sample calendar
        calendar_events = []
        for day, events in sample_calendar.items():
            for event in events:
                calendar_events.append({
                    "Day": day,
                    "Start": event['start'],
                    "End": event['end'],
                    "Activity": event['name']
                })
        
        # Display as a table
        calendar_df = pd.DataFrame(calendar_events)
        st.dataframe(calendar_df, use_container_width=True)
        
        st.markdown("""
        **This sample calendar includes:**
        
        - Regular class times for all three courses
        - Study group sessions
        - Part-time job shifts
        - Extracurricular activities
        - Social commitments
        
        The system will automatically work around these commitments when scheduling study time.
        """)

# Add a footer
st.markdown("""
---
### Syllabus Analyzer & Assignment Scheduler
Built with Streamlit • Made for students, by students
""")