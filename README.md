# ResumeMatch AI

An NLP-powered Resume–Job Description Matching System that helps job seekers understand how their resumes align with target job descriptions.

## Overview

ResumeMatch AI analyzes a resume and a job description to identify relevant skills, highlight potential skill gaps, and estimate textual similarity.

This is a candidate-support tool, not an automated hiring or rejection system.

## Features

- Upload resumes in PDF format
- Extract resume text using PyMuPDF
- Identify skills using a predefined skill taxonomy
- Calculate TF-IDF cosine similarity
- Estimate skill coverage
- Display matched and missing skills
- Visualize matching results with interactive charts
- Download an analysis report

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- PyMuPDF
- Pandas
- Plotly
- Natural Language Processing (NLP)

## How It Works

1. Upload a resume in PDF format.
2. Paste the target job description.
3. Extract resume text and identify relevant skills.
4. Compare the resume with the job description.
5. Review similarity, skill coverage, and potential skill gaps.
6. Download the analysis report.

## Run Locally

Install the dependencies:

```bash
pip install -r requirements.txt
