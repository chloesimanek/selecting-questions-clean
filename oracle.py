import numpy as np
import torch
import json
import pandas as pd
from copy import deepcopy
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sigmoid(x):
    """Standard sigmoid function"""
    z = np.exp(-x)
    return 1 / (1 + z)

def proba(theta, d):
    """
    Calculate probability of correct answer using IRT model
    
    Args:
        theta: Student ability for the skill
        d: Question difficulty for the skill
        
    Returns:
        Probability of correct answer
    """
    return sigmoid(theta - d)

def generate_synthetic_data(n_students=6400, n_questions=948, n_skills=1):
    """
    Generate synthetic data with single-skill questions
    
    Args:
        n_students: Number of students to generate
        n_questions: Number of questions to generate
        n_skills: Total number of skills
        
    Returns:
        Dictionary containing synthetic data
    """
    np.random.seed(8080)
    
    # Generate student abilities for each skill
    student_thetas = np.random.normal(0, 1, (n_students, n_skills))
    
    # Add correlation between skills (students good at one skill tend to be good at others)
    skill_correlation = 0.5
    shared_ability = np.random.normal(0, 1, n_students)
    for s in range(n_skills):
        student_thetas[:, s] = skill_correlation * shared_ability + (1 - skill_correlation) * student_thetas[:, s]
    
    # Generate base difficulties for each skill
    skill_difficulties = []
    for i in range(n_skills):
        mu = np.random.uniform(-0.5, 0.5)  # Skill-specific mean difficulty
        sigma = np.random.uniform(0.8, 1.2)  # Skill-specific spread
        skill_difficulties.append((mu, sigma))
    
    # Generate question data
    question_data = []
    
    # Assign skills to questions roughly equally
    questions_per_skill = n_questions // n_skills
    remaining = n_questions % n_skills
    
    skill_counts = [questions_per_skill] * n_skills
    for i in range(remaining):
        skill_counts[i] += 1
    
    q_id = 0
    for skill_id in range(n_skills):
        mu, sigma = skill_difficulties[skill_id]
        
        for _ in range(skill_counts[skill_id]):
            # Generate difficulty from the skill's distribution
            difficulty = np.random.normal(mu, sigma)
            
            question_data.append({
                'id': q_id, 
                'difficulty': difficulty,
                'skill': skill_id
            })
            q_id += 1
    
    # Generate response data in required format for BOBCAT - JSON
    train_data = []
    
    for s in range(n_students):
        student_abilities = student_thetas[s]
        
        # Randomly determine how many questions this student answers (20-200) 
        n_responses = np.random.randint(20, min(200, n_questions))
        
        # Randomly select questions w/o replacement
        q_indices = np.random.choice(n_questions, n_responses, replace=False)
        
        # Generate responses
        q_ids = q_indices.tolist()
        labels = []
        
        for q in q_indices:
            q_data = question_data[q]
            skill_id = q_data['skill']
            difficulty = q_data['difficulty']
            
            # Calculate probability of correct answer
            p = proba(student_abilities[skill_id], difficulty)
            correct = np.random.random() < p
            labels.append(1 if correct else 0)
        
        # Create student data entry
        student_data = {
            'user_id': int(s),
            'q_ids': q_ids,
            'labels': labels
        }
        
        train_data.append(student_data)
    
    # Detailed response data for analysis
    detailed_responses = {}
    for s in range(n_students):
        student_abilities = student_thetas[s]
        student_responses = {}
        
        for q in range(n_questions):
            q_data = question_data[q]
            skill_id = q_data['skill']
            difficulty = q_data['difficulty']
            
            # Calculate probability of correct answer
            p = proba(student_abilities[skill_id], difficulty)
            correct = np.random.random() < p
            
            student_responses[q] = {
                'correct': 1 if correct else 0,
                'probability': float(p),
                'skill': skill_id
            }
        
        detailed_responses[s] = student_responses
    
    # Return complete synthetic data
    return {
        'student_abilities': student_thetas,
        'question_data': question_data,
        'skill_difficulties': skill_difficulties,
        'train_data': train_data,
        'detailed_responses': detailed_responses
    }

def save_synthetic_data(synthetic_data, output_dir="data/synthetic"):
    """
    Save synthetic data to files
    
    Args:
        synthetic_data: Dictionary containing synthetic data
        output_dir: Directory to save files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON formatted data for BOBCAT
    json_path = os.path.join(output_dir, "synthetic_train_data.json")
    with open(json_path, 'w') as f:
        json.dump(synthetic_data['train_data'], f)
    
    # Save student parameters
    student_abilities = synthetic_data['student_abilities']
    n_students, n_skills = student_abilities.shape
    
    student_data = []
    for s in range(n_students):
        student_row = {'student_id': s}
        for skill in range(n_skills):
            student_row[f'skill_{skill}'] = student_abilities[s, skill]
        student_data.append(student_row)
    
    student_df = pd.DataFrame(student_data)
    student_df.to_csv(os.path.join(output_dir, "synthetic_students.csv"), index=False)
    
    # Save question parameters
    question_rows = []
    for q_data in synthetic_data['question_data']:
        question_rows.append({
            'question_id': q_data['id'],
            'skill': q_data['skill'],
            'difficulty': q_data['difficulty']
        })
    
    question_df = pd.DataFrame(question_rows)
    question_df.to_csv(os.path.join(output_dir, "synthetic_questions.csv"), index=False)
    
    # Save skill parameters
    skill_df = pd.DataFrame(synthetic_data['skill_difficulties'], 
                          columns=['mean_difficulty', 'std_difficulty'])
    skill_df['skill_id'] = range(len(synthetic_data['skill_difficulties']))
    skill_df.to_csv(os.path.join(output_dir, "synthetic_skills.csv"), index=False)
    
    # Save model parameters
    model_params = {
        'n_students': len(synthetic_data['student_abilities']),
        'n_questions': len(synthetic_data['question_data']),
        'n_skills': synthetic_data['student_abilities'].shape[1],
        'avg_ability': float(np.mean(synthetic_data['student_abilities'])),
        'avg_difficulty': float(np.mean([q['difficulty'] for q in synthetic_data['question_data']]))
    }
    
    with open(os.path.join(output_dir, "synthetic_model_params.json"), 'w') as f:
        json.dump(model_params, f)
    
    print(f"Synthetic data saved to {output_dir}")
    return json_path

def oracle_questions(student_data, question_data, n_questions=20):
    """
    Creates a bank of optimal questions for each student based on their abilities.
    This function is run only once to precompute the best questions.
    
    Args:
        student_data: Dictionary mapping student IDs to their ability estimates across skills
        question_data: List of question parameters (difficulty, skill ID, etc.)
        n_questions: Number of questions to select per student
        
    Returns:
        Dictionary mapping student IDs to their optimal question sets
    """
    

if __name__ == "__main__":
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(
        n_students=6400, 
        n_questions=948, 
        n_skills=10
    )
    
    # Save the data
    output_path = save_synthetic_data(synthetic_data)
    
    print(f"Generated synthetic data with:")
    print(f"  - {len(synthetic_data['student_abilities'])} students")
    print(f"  - {len(synthetic_data['question_data'])} questions")
    print(f"  - {synthetic_data['student_abilities'].shape[1]} skills")
    print(f"  - Average student ability: {np.mean(synthetic_data['student_abilities']):.4f}") # to 4 dp