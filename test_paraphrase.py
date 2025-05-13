#!/usr/bin/env python3
# test_paraphrase.py - Direct testing of paraphrasing functions

import os
import sys
import time
from dotenv import load_dotenv
import argparse

# Import functions from app.py (assuming app.py is in the same directory)
from app import (
    modern_concise_paraphrase, 
    specialization_emphasis_paraphrase,
    technical_distinction_paraphrase,
    role_highlight_paraphrase,
    descriptive_detailed_paraphrase,
    ensure_paraphrase_results
)

# Load environment variables
load_dotenv()

def test_paraphrasing(resume_text, style=None):
    """Test a specific paraphrasing style or all styles"""
    print(f"Testing paraphrasing with resume text of {len(resume_text)} characters")
    
    # Map of styles to functions
    style_functions = {
        "modern": modern_concise_paraphrase,
        "specialization": specialization_emphasis_paraphrase,
        "technical": technical_distinction_paraphrase,
        "role": role_highlight_paraphrase,
        "descriptive": descriptive_detailed_paraphrase
    }
    
    if style and style in style_functions:
        # Test only the specified style
        print(f"\n=== Testing {style} style ===")
        start_time = time.time()
        result = style_functions[style](resume_text)
        elapsed = time.time() - start_time
        
        if result:
            print(f"SUCCESS ({elapsed:.2f}s): Got result with {len(result)} characters")
            print("\nSample of result:")
            print(result[:300] + "..." if len(result) > 300 else result)
        else:
            print(f"FAILED ({elapsed:.2f}s): No result returned")
            
            # Try fallback
            print("Trying fallback method...")
            start_time = time.time()
            fallback = ensure_paraphrase_results(resume_text)
            elapsed = time.time() - start_time
            
            if fallback and "text" in fallback:
                print(f"FALLBACK SUCCESS ({elapsed:.2f}s): Got result with {len(fallback['text'])} characters")
                print("\nSample of fallback result:")
                print(fallback['text'][:300] + "..." if len(fallback['text']) > 300 else fallback['text'])
            else:
                print(f"FALLBACK FAILED ({elapsed:.2f}s): No result returned")
    else:
        # Test all styles
        for name, func in style_functions.items():
            print(f"\n=== Testing {name} style ===")
            start_time = time.time()
            result = func(resume_text)
            elapsed = time.time() - start_time
            
            if result:
                print(f"SUCCESS ({elapsed:.2f}s): Got result with {len(result)} characters")
                print("\nSample of result:")
                print(result[:200] + "..." if len(result) > 200 else result)
            else:
                print(f"FAILED ({elapsed:.2f}s): No result returned")

def main():
    parser = argparse.ArgumentParser(description="Test resume paraphrasing")
    parser.add_argument("--file", "-f", help="Path to resume text file")
    parser.add_argument("--style", "-s", help="Paraphrasing style (modern, specialization, technical, role, descriptive)")
    args = parser.parse_args()
    
    resume_text = ""
    
    if args.file:
        try:
            with open(args.file, 'r') as f:
                resume_text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    else:
        # Use sample resume text
        resume_text = """SOFTWARE ENGINEER
Professional Summary
Innovative Software Engineer with 5+ years of experience in full-stack development, specializing in cloud-native applications and microservices architecture. Proven track record of delivering scalable solutions that drive business growth and enhance user experience.

Skills
• Programming: Java, Python, JavaScript, TypeScript
• Frameworks: Spring Boot, React, Angular, Node.js
• Cloud: AWS (EC2, S3, Lambda), Docker, Kubernetes
• Database: MySQL, MongoDB, PostgreSQL
• DevOps: Jenkins, GitHub Actions, Terraform
• Testing: JUnit, Mockito, Jest

Experience
Senior Software Engineer | TechCorp Inc. | Jan 2020 - Present
• Led a team of 5 developers to redesign the company's flagship product, resulting in a 40% increase in user engagement
• Implemented microservices architecture using Spring Boot and Docker, improving system scalability and reducing deployment time by 60%
• Developed and maintained CI/CD pipelines using Jenkins and GitHub Actions
• Conducted code reviews and mentored junior developers, improving team productivity by 25%
• Collaborated with product managers to define technical requirements and timelines

Software Engineer | DataSystems LLC | Mar 2017 - Dec 2019
• Designed and implemented RESTful APIs using Node.js and Express, supporting mobile and web applications
• Developed front-end components using React and Redux, enhancing user experience
• Optimized database queries, reducing response time by 30%
• Participated in Agile development processes, including daily stand-ups and sprint planning

Education
Bachelor of Science in Computer Science
University of Technology | Graduated: May 2017
• GPA: 3.8/4.0
• Relevant Coursework: Data Structures, Algorithms, Database Systems, Web Development"""
    
    if not resume_text:
        print("No resume text provided")
        sys.exit(1)
    
    test_paraphrasing(resume_text, args.style)

if __name__ == "__main__":
    main()