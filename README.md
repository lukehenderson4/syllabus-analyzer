# Syllabus Analyzer & Assignment Scheduler

This tool helps students analyze course syllabi, extract assignment information, and create optimized study schedules that work around existing commitments.


## Features

- **Multi-Course Syllabus Analysis**: Upload or paste syllabi from multiple courses
- **Automatic Assignment Extraction**: Identifies due dates, page counts, and word counts
- **Calendar Integration**: Import your existing calendar to automatically block off busy times
- **Smart Scheduling**: Creates an optimized study schedule that works around your existing commitments
- **Workload Visualization**: Visual analysis of workload distribution across courses
- **Calendar View**: Interactive calendar showing both study tasks and existing commitments
- **Export Options**: Export your schedule to CSV or iCalendar format

## Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **ICS**: Calendar file parsing and generation
- **PyPDF2/python-docx**: Document parsing

## Installation

1. Clone this repository:
```bash
git clone https://github.com/lukehenderson4/syllabus-analyzer.git
cd syllabus-analyzer
```

2. Create a virtual environment:
```bash
python -m venv venv # On Mac, it may be python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt # On Mac, it may be pip3 install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run app2.py
```

2. Open your web browser and go to http://localhost:8501

3. **Add Course Syllabi**:
   - Use the "Add Syllabi" tab
   - Paste syllabus text or upload syllabus files (TXT, PDF, DOC/DOCX)
   - The system will automatically extract assignments, due dates, and workload information

4. **Review Analysis**:
   - Go to the "Analysis" tab to see workload distribution
   - View the timeline of assignments across the semester

5. **Set Up Your Schedule**:
   - Configure your time parameters in the sidebar
   - Set available time slots or import your calendar
   - Go to the "Schedule" tab and generate your study schedule

6. **Export Your Schedule**:
   - Export to CSV for spreadsheet applications
   - Export to iCalendar format for Google Calendar, Apple Calendar, etc.

## Sample Data

For a quick demonstration, click the "Load Sample Data" button in the sidebar. This will:

1. Load three sample syllabi for Spring 2025 (CS101, ENG201, MATH150)
2. Import a sample weekly calendar with classes, work, and activities
3. Generate available study time slots around these commitments

## Time Parameters

Customize time estimates based on your own reading and writing speeds:
- **Reading Rate**: Minutes it takes to read 10 pages
- **Writing Rate**: Minutes it takes to write 100 words

## File Formats

The following file formats are supported for syllabus upload:
- Plain text (.txt)
- PDF documents (.pdf)
- Word documents (.doc, .docx)

For calendar import, iCalendar format (.ics) is supported, which can be exported from:
- Google Calendar
- Apple Calendar
- Microsoft Outlook
- And most other calendar applications

## Future Improvements

Given additional time, this is how I would enhance the product:
- Use machine learning like spaCy natural language processing to train the model on different types of syllabi so it could better extract information
- Added functionality to distinguish different reading and writing times for different courses
- Mobile app integration
- Progress tracking with notifications
- Integration with learning management systems (Canvas, Blackboard)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Built with Python and Streamlit • Made for students, by students
