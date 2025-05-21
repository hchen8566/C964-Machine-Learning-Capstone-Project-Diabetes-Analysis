Competencies
981.1.1 : Capstone

The learner integrates and synthesizes competencies from across the degree program and thereby demonstrates the ability to participate in and contribute value to the chosen professional field.

Introduction
Before starting this task, ensure that your capstone project has been approved by your instructor.

In this task, you will design, develop, and implement the capstone project approved by your instructor in Task 1.

Note: Your work for this task will not be evaluated until you have successfully passed Task 1.
Requirements
Your submission must be your original work. No more than a combined total of 30% of the submission and no more than a 10% match to any one individual source can be directly quoted or closely paraphrased from sources, even if cited correctly. The similarity report that is provided when you submit your task can be used as a guide.

 

You must use the rubric to direct the creation of your submission because it provides detailed criteria that will be used to evaluate your work. Each requirement below may be evaluated by more than one rubric aspect. The rubric aspect titles may contain hyperlinks to relevant portions of the course.

 

Tasks may not be submitted as cloud links, such as links to Google Docs, Google Slides, OneDrive, etc., unless specified in the task requirements. All other submissions must be file types that are uploaded and submitted as attachments (e.g., .docx, .pdf, .ppt).


A.  Create a letter of transmittal and a project proposal to convince senior, nontechnical managers and executives to implement your data product approved in Task 1. The proposal should include each of the following:

•   a summary of the problem

•   a description of how the data product benefits the customer and supports the decision-making process

•   an outline of the data product

•   a description of the data that will be used to construct the data product

•   the objectives and hypotheses of the project

•   an outline of the project methodology

•   funding requirements

•   the impact of the solution on stakeholders

•   ethical and legal considerations and precautions that will be used when working with and communicating about sensitive data

•   your expertise relevant to the solution you propose

 
    Note: Expertise described here could be real or hypothetical to fit the project topic you have created.

 

B.  Write an executive summary directed to IT professionals that addresses each of the following requirements:

•   the decision support problem or opportunity you are solving for

•   a description of the customers and why this product will fulfill their needs

•   existing gaps in the data products you are replacing or modifying (if applicable)

•   the data available or the data that needs to be collected to support the data product lifecycle

•   the methodology you use to guide and support the data product design and development

•   deliverables associated with the design and development of the data product

•   the plan for implementation of your data product, including the anticipated outcomes from this development

•   the methods for validating and verifying that the developed data product meets the requirements and, subsequently, the needs of the customers

•   the programming environments and any related costs, as well as the human resources that are necessary to execute each phase in the development of the data product

•   a projected timeline, including milestones, start and end dates, duration for each milestone, dependencies, and resources assigned to each task

 

C.  Design and develop your fully functional data product that addresses your identified business problem or organizational need from part A. Include each of the following attributes, as they are the minimum required elements for the product:

•   one descriptive method and one nondescriptive (predictive or prescriptive) method

•   collected or available datasets

•   decision support functionality

•   ability to support featurizing, parsing, cleaning, and wrangling datasets

•   methods and algorithms supporting data exploration and preparation

•   data visualization functionalities for data exploration and inspection

•   implementation of interactive queries

•   implementation of machine-learning methods and algorithms

•   functionalities to evaluate the accuracy of the data product

•   industry-appropriate security features

•   tools to monitor and maintain the product

•   a user-friendly, functional dashboard that includes three visualization types

 

D.  Create each of the following forms of documentation for the product you have developed:

•   a business vision or business requirements document

•   raw and cleaned datasets with the code and executable files used to scrape and clean data (if applicable)

•   code used to perform the analysis of the data and construct a descriptive, predictive, or prescriptive data product

•   assessment of the hypotheses for acceptance or rejection

•   visualizations and elements of effective storytelling supporting the data exploration and preparation, data analysis, and data summary, including the phenomenon and its detection

•   assessment of the product’s accuracy 

•   the results from the data product testing, revisions, and optimization based on the provided plans, including screenshots

•   source code and executable file(s)

•   a quick-start guide summarizing the steps necessary to install and use the product

 

E.  Acknowledge sources, using in-text citations and references, for content that is quoted, paraphrased, or summarized.

 

F.  Demonstrate professional communication in the content and presentation of your submission.

File Restrictions
File name may contain only letters, numbers, spaces, and these symbols: ! - _ . * ' ( )
File size limit: 200 MB
File types allowed: doc, docx, rtf, xls, xlsx, ppt, pptx, odt, pdf, csv, txt, qt, mov, mpg, avi, mp3, wav, mp4, wma, flv, asf, mpeg, wmv, m4v, svg, tif, tiff, jpeg, jpg, gif, png, zip, rar, tar, 7z

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

REQUIREMENTS TO RUN THE APP NATIVELY

You must have Python 3.8+ installed.

Required Python libraries:
- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- joblib

These are listed in `requirements.txt` and can be installed automatically.

---

HOW TO RUN THE APP

1. Install Python if you don’t already have it: https://www.python.org/downloads/

2. Download or unzip the project folder. You should see:

   - user_app.py 
   - train_random_forest.py 
   - diabetes_model.pkl  
   - diabetes.csv
   - requirements.txt  
   - README.txt

3. Open a terminal or command prompt and navigate into the folder

4. Use command 'pip install -r requirements.txt'

5. Launch the app with the command 'streamlit run app.py'

6. App should auto open in your preferred browser via streamlit

---

NOTES

- This is an educational tool built for a machine learning capstone project.
- It is not intended for clinical or diagnostic use.
- The model was trained using the publicly available Pima Indian Diabetes dataset.