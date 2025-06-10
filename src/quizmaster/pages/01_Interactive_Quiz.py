import streamlit as st
import sys
import os
import re
from datetime import datetime

# Add parent directory to path to enable relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from quizmaster.database_manager import DatabaseManager
from quizmaster.insights_generator import generate_insights

def show_quiz_page():
    st.set_page_config(
        page_title="Interactive Quiz - QuizMaster Pro", 
        page_icon="🎓", 
        layout="wide"
    )
    st.title("🎓 Interactive Quiz")

    if 'quiz_data' not in st.session_state or not st.session_state.quiz_data or not st.session_state.quiz_data.get('questions'):
        st.info("👋 Please generate a quiz on the 'Setup & Generation' page first.")
        st.page_link("streamlit_app.py", label="Go to Setup", icon="⚙️")
        return

    quiz_data = st.session_state.quiz_data
    questions = quiz_data['questions']
    
    # Retrieve stored difficulty, default if not found (should be set by main app)
    difficulty = quiz_data.get('config_difficulty', 'Medium') 

    # Ensure quiz state variables are initialized (though main app should do this)
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {} # Should be initialized per quiz
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False

    # Quiz navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.session_state.current_question > 0:
            if st.button("⬅️ Previous"):
                st.session_state.current_question -= 1
                st.rerun()
        else:
            st.write("") # Placeholder for alignment

    with col2:
        st.write(f"Question {st.session_state.current_question + 1} of {len(questions)}")
        progress_value = (st.session_state.current_question + 1) / len(questions) if questions else 0
        st.progress(progress_value)
    
    with col3:
        if st.session_state.current_question < len(questions) - 1:
            if st.button("Next ➡️"):
                st.session_state.current_question += 1
                st.rerun()
        else:
            st.write("") # Placeholder for alignment

    # Display current question
    current_q = questions[st.session_state.current_question]
    question_key = f"q_{st.session_state.current_question}"
    
    st.subheader(f"Question {st.session_state.current_question + 1}")
    st.markdown(f"**{current_q['text']}**")
    st.markdown(f"*Type: {current_q['type']} | Difficulty: {difficulty}*")
    
    # Question-specific input
    user_answer = None
    
    if current_q['type'] == "Multiple Choice":
        options = [opt['text'] for opt in current_q['options']]
        # Get previous answer if exists for radio button
        previous_answer_index = None
        if question_key in st.session_state.user_answers:
            try:
                previous_answer_index = options.index(st.session_state.user_answers[question_key])
            except ValueError:
                previous_answer_index = None # Answer not in current options

        user_answer = st.radio(
            "Select your answer:",
            options,
            key=question_key,
            index=previous_answer_index
        )
        
    elif current_q['type'] == "Open-Ended":
        user_answer = st.text_area(
            "Your answer:",
            value=st.session_state.user_answers.get(question_key, ""),
            key=question_key,
            height=100
        )
        
    elif current_q['type'] == "True/False":
        previous_answer_tf = st.session_state.user_answers.get(question_key)
        tf_options = ["True", "False"]
        previous_answer_index_tf = None
        if previous_answer_tf in tf_options:
            previous_answer_index_tf = tf_options.index(previous_answer_tf)

        user_answer = st.radio(
            "Select:",
            tf_options,
            key=question_key,
            index=previous_answer_index_tf
        )
        
    elif current_q['type'] == "Fill-in-the-Blank":
        user_answer = st.text_input(
            "Fill in the blank:",
            value=st.session_state.user_answers.get(question_key, ""),
            key=question_key
        )
    
    # Store user answer
    if user_answer is not None: # Check if widget returned a value
        # For text_area and text_input, an empty string is a valid input from the widget
        # For radio, None means no selection yet.
        if current_q['type'] in ["Open-Ended", "Fill-in-the-Blank"] or user_answer: 
            st.session_state.user_answers[question_key] = user_answer
    
    # Show answer and explanation
    if st.button("💡 Show Answer", key=f"show_{question_key}"):
        st.info("**Correct Answer:**")
        correct_option_text = "" # For MCQ
        if current_q['type'] == "Multiple Choice":
            correct_letter = current_q['answer']
            for opt in current_q['options']:
                if opt['letter'] == correct_letter:
                    correct_option_text = opt['text']
                    break
            st.write(f"{correct_letter}: {correct_option_text}")
        else:
            st.write(str(current_q['answer'])) # Ensure boolean True/False are strings for display consistency
        
        if 'explanation' in current_q and current_q['explanation']:
            st.success("**Explanation:**")
            st.write(current_q['explanation'])
        
        # Show user's answer if provided and compare
        if question_key in st.session_state.user_answers:
            user_ans_display = st.session_state.user_answers[question_key]
            if user_ans_display or isinstance(user_ans_display, str): # Check if answer is not None or empty string for some types
                is_correct = False
                if current_q['type'] == "Multiple Choice":
                    is_correct = user_ans_display == correct_option_text
                elif current_q['type'] == "True/False":
                    # Compare boolean answer with string representation from radio
                    is_correct = str(user_ans_display).lower() == str(current_q['answer']).lower()
                elif current_q['type'] == "Fill-in-the-Blank":
                    is_correct = user_ans_display.strip().lower() == str(current_q['answer']).strip().lower()
                # Open-Ended questions are not auto-graded here
                
                if current_q['type'] != "Open-Ended":
                    if is_correct:
                        st.success(f"✅ Your answer '{user_ans_display}' is correct!")
                    else:
                        st.error(f"❌ Your answer '{user_ans_display}' is incorrect.")
                else:
                    st.info(f"Your answer: '{user_ans_display}'")


    # Final results
    if st.session_state.current_question == len(questions) - 1:
        st.divider()
        if st.button("📊 Show Final Results", type="primary"):
            st.session_state.show_results = True
            st.rerun() # Rerun to display results section

    if st.session_state.show_results:
        st.divider()
        st.header("🏆 Quiz Results")
        
        answered_questions = 0
        correct_answers = 0
        gradable_questions = 0
        
        # Prepare data for saving
        results = {
            'total_questions': len(questions),
            'answered_questions': 0,
            'correct_answers': 0,
            'completion_rate': 0,
            'score': 0
        }
        
        # Create list of questions for insight generation
        insight_questions = []
        
        # First pass: collect data for insights and results
        for i, q_data in enumerate(questions):
            q_key = f"q_{i}"
            user_ans_val = st.session_state.user_answers.get(q_key)
            insight_q = {
                'text': q_data['text'],
                'type': q_data['type'],
                'user_answer': user_ans_val,
                'correct_answer': q_data['answer'],
                'is_correct': False
            }
            
            if user_ans_val is not None:
                if not (isinstance(user_ans_val, str) and not user_ans_val.strip()):
                    answered_questions += 1
                    results['answered_questions'] = answered_questions

            # Check correctness for auto-gradable types
            if q_data['type'] != "Open-Ended" and user_ans_val is not None:
                gradable_questions += 1
                q_correct_answer = q_data['answer']
                is_q_correct = False
                if q_data['type'] == "Multiple Choice":
                    correct_letter_val = q_data['answer']
                    user_choice_text = user_ans_val
                    actual_correct_text = ""
                    
                    # Debug logging
                    print(f"Question {i}: Checking Multiple Choice answer")
                    print(f"User answer: '{user_choice_text}'")
                    print(f"Correct letter: '{correct_letter_val}'")
                    
                    for opt_val in q_data['options']:
                        if opt_val['letter'] == correct_letter_val:
                            actual_correct_text = opt_val['text']
                            print(f"Correct text: '{actual_correct_text}'")
                            break
                    
                    # More flexible comparison - strip whitespace and ignore case
                    if user_choice_text and actual_correct_text and user_choice_text.strip().lower() == actual_correct_text.strip().lower():
                        is_q_correct = True
                        correct_answers += 1
                        print(f"Question {i}: CORRECT")
                    else:
                        print(f"Question {i}: INCORRECT")
                elif q_data['type'] == "True/False":
                    # Debug logging
                    print(f"Question {i}: Checking True/False answer")
                    print(f"User answer: '{user_ans_val}'")
                    print(f"Correct answer: '{q_correct_answer}'")
                    
                    # More flexible comparison
                    user_answer_normalized = str(user_ans_val).strip().lower()
                    correct_answer_normalized = str(q_correct_answer).strip().lower()
                    
                    # Handle various forms of true/false answers
                    if user_answer_normalized in ['true', 't', 'yes', 'y', '1'] and correct_answer_normalized in ['true', 't', 'yes', 'y', '1']:
                        is_q_correct = True
                        correct_answers += 1
                        print(f"Question {i}: CORRECT")
                    elif user_answer_normalized in ['false', 'f', 'no', 'n', '0'] and correct_answer_normalized in ['false', 'f', 'no', 'n', '0']:
                        is_q_correct = True
                        correct_answers += 1
                        print(f"Question {i}: CORRECT")
                    else:
                        # Direct string comparison as fallback
                        if user_answer_normalized == correct_answer_normalized:
                            is_q_correct = True
                            correct_answers += 1
                            print(f"Question {i}: CORRECT")
                        else:
                            print(f"Question {i}: INCORRECT")
                            
                elif q_data['type'] == "Fill-in-the-Blank":
                    # Debug logging
                    print(f"Question {i}: Checking Fill-in-the-Blank answer")
                    print(f"User answer: '{user_ans_val}'")
                    print(f"Correct answer: '{q_correct_answer}'")
                    
                    # More flexible comparison - normalize whitespace, punctuation, and case
                    user_answer_normalized = re.sub(r'[^\w\s]', '', str(user_ans_val)).strip().lower()
                    correct_answer_normalized = re.sub(r'[^\w\s]', '', str(q_correct_answer)).strip().lower()
                    
                    if user_answer_normalized == correct_answer_normalized:
                        is_q_correct = True
                        correct_answers += 1
                        print(f"Question {i}: CORRECT")
                    else:
                        print(f"Question {i}: INCORRECT")
                
                insight_q['is_correct'] = is_q_correct
                
            insight_questions.append(insight_q)
        
        results['correct_answers'] = correct_answers
        completion_rate = (answered_questions / len(questions)) * 100 if questions else 0
        score = (correct_answers / gradable_questions) * 100 if gradable_questions > 0 else 0
        results['completion_rate'] = completion_rate
        results['score'] = score
        
        # Save results to database
        db = DatabaseManager()
        session_id = db.save_quiz_session(quiz_data, results)
        db.save_quiz_answers(session_id, questions, st.session_state.user_answers)
        
        # Generate insights
        model_used_for_quiz = quiz_data.get('model_used', 'llama3') # Default to llama3 if not found
        insights = generate_insights(insight_questions, model_used_for_quiz)
        db.save_quiz_report(session_id, insights)
        
        # Display results
        col1_res, col2_res, col3_res = st.columns(3)
        with col1_res:
            st.metric("Questions Answered", f"{answered_questions}/{len(questions)}")
        with col2_res:
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
        # Only show score if there are gradable questions
        if gradable_questions > 0:
            with col3_res:
                 st.metric("Score (Auto-graded)", f"{score:.1f}% ({correct_answers}/{gradable_questions})")
        else:
            with col3_res:
                st.write("No auto-gradable questions.")

        st.metric("Model Used for Quiz Generation", quiz_data['model_used'])
        
        # Display insights with better formatting
        st.divider()
        st.header("📊 Knowledge Insights")
        
        # Check if insights contain markdown headings
        if "# " in insights or "## " in insights:
            # If insights already have markdown formatting, display as is
            st.markdown(insights)
        else:
            # Otherwise, display as regular text
            st.write(insights)
        
        # Display selected concepts if available
        if 'selected_concepts' in quiz_data and quiz_data['selected_concepts']:
            st.divider()
            st.header("🎯 Concepts Tested")
            st.write("This quiz was focused on the following concepts:")
            for concept in quiz_data['selected_concepts']:
                st.markdown(f"• {concept}")
        
        st.divider()
        
        # Save quiz section
        with st.expander("💾 Save This Quiz"):
            quiz_title = st.text_input("Quiz Title", 
                                     value=f"{quiz_data['model_used']} Quiz - {datetime.now().strftime('%Y-%m-%d')}")
            quiz_description = st.text_area("Description (optional)")
            tags = st.text_input("Tags (comma separated, optional)")
            
            if st.button("💾 Save Quiz", type="primary"):
                if not quiz_title.strip():
                    st.error("Please enter a title for the quiz")
                else:
                    tags_list = [t.strip() for t in tags.split(",")] if tags else []
                    quiz_id = db.save_quiz(
                        title=quiz_title,
                        quiz_data=quiz_data,
                        description=quiz_description,
                        tags=tags_list
                    )
                    st.success(f"Quiz saved successfully! (ID: {quiz_id})")
        
        if st.button("🔄 Take Quiz Again"):
            st.session_state.current_question = 0
            st.session_state.user_answers = {}
            st.session_state.show_results = False
            st.rerun()
        
        if st.button("🏠 Back to Setup"):
            st.switch_page("streamlit_app.py")
            
        if st.button("📚 View Saved Quizzes"):
            st.switch_page("pages/02_Saved_Quizzes.py")


if __name__ == "__main__":
    show_quiz_page()
