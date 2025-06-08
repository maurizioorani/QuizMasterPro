import streamlit as st

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
        
        for i, q_data in enumerate(questions):
            q_key = f"q_{i}"
            user_ans_val = st.session_state.user_answers.get(q_key)

            if user_ans_val is not None: # Considers any answer attempt
                 if not (isinstance(user_ans_val, str) and not user_ans_val.strip()): # Not an empty string
                    answered_questions +=1

            # Check correctness for auto-gradable types
            if q_data['type'] != "Open-Ended" and user_ans_val is not None:
                q_correct_answer = q_data['answer']
                is_q_correct = False
                if q_data['type'] == "Multiple Choice":
                    correct_letter_val = q_data['answer']
                    user_choice_text = user_ans_val
                    actual_correct_text = ""
                    for opt_val in q_data['options']:
                        if opt_val['letter'] == correct_letter_val:
                            actual_correct_text = opt_val['text']
                            break
                    if user_choice_text == actual_correct_text:
                        is_q_correct = True
                elif q_data['type'] == "True/False":
                    if str(user_ans_val).lower() == str(q_correct_answer).lower():
                        is_q_correct = True
                elif q_data['type'] == "Fill-in-the-Blank":
                    if str(user_ans_val).strip().lower() == str(q_correct_answer).strip().lower():
                        is_q_correct = True
                
                if is_q_correct:
                    correct_answers +=1

        completion_rate = (answered_questions / len(questions)) * 100 if questions else 0
        score = (correct_answers / len([q for q in questions if q['type'] != "Open-Ended"])) * 100 if len([q for q in questions if q['type'] != "Open-Ended"]) > 0 else 0

        col1_res, col2_res, col3_res = st.columns(3)
        with col1_res:
            st.metric("Questions Answered", f"{answered_questions}/{len(questions)}")
        with col2_res:
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
        # Only show score if there are gradable questions
        gradable_questions = sum(1 for q in questions if q['type'] != "Open-Ended")
        if gradable_questions > 0:
            with col3_res:
                 st.metric("Score (Auto-graded)", f"{score:.1f}% ({correct_answers}/{gradable_questions})")
        else:
            with col3_res:
                st.write("No auto-gradable questions.")

        st.metric("Model Used for Quiz Generation", quiz_data['model_used'])
        
        if st.button("🔄 Take Quiz Again"):
            st.session_state.current_question = 0
            st.session_state.user_answers = {}
            st.session_state.show_results = False
            st.rerun()
        
        if st.button("🏠 Back to Setup"):
            # Potentially clear quiz_data if desired, or keep it
            # st.session_state.quiz_data = None 
            st.switch_page("streamlit_app.py")


if __name__ == "__main__":
    show_quiz_page()