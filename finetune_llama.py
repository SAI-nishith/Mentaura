import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys

# Setup logging for tracking errors and important events
logging.basicConfig(
    filename="training.log",
    filemode="a",  
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Load the MBTI tokenizer and model
mbti_tokenizer = AutoTokenizer.from_pretrained("theta/mbti-career")
mbti_model = AutoModelForSequenceClassification.from_pretrained("theta/mbti-career")

# Function to classify personality based on input text using MBTI model
def classify_personality(input_text):
    """
    Classifies personality type based on the input text using the MBTI model.
    
    Args:
    input_text (str): A description of a person's characteristics or preferences.
    
    Returns:
    personality_class (int): Predicted personality class index.
    class_probabilities (torch.Tensor): The probabilities for each personality type.
    """
    # Tokenize the input text and convert it to tensor format for the model
    model_inputs = mbti_tokenizer(input_text, return_tensors="pt")
    
    # Pass the tokenized input through the model to get raw predictions (logits)
    model_outputs = mbti_model(**model_inputs)
    logits = model_outputs.logits
    
    # Apply softmax to convert logits to class probabilities
    class_probabilities = torch.softmax(logits, dim=-1)
    
    # Find the index of the class with the highest probability
    personality_class = torch.argmax(class_probabilities, dim=-1).item()
    
    return personality_class, class_probabilities

# Function to process open-ended answers and map them to RIASEC job types
def process_open_ended_responses(responses):
    """
    Processes multiple open-ended responses to classify personality and suggest careers.
    
    Args:
    responses (list of str): A list of open-ended answers provided by the user.
    
    Returns:
    top_riasec_types (list of str): Top two RIASEC types based on responses.
    job_recommendations (str): Job recommendations based on the top RIASEC types.
    """
    try:
        # Initialize a dictionary to track RIASEC type scores
        riasec_scores = {
            'Realistic': 0, 
            'Investigative': 0, 
            'Artistic': 0,
            'Social': 0, 
            'Enterprising': 0, 
            'Conventional': 0
        }

        # Example mapping of MBTI personality class indices to RIASEC types
        mbti_to_riasec_mapping = {
            0: 'Realistic',       # Example: Logical and technical individuals
            1: 'Investigative',    # Example: Analytical or curious individuals
            2: 'Artistic',         # Example: Creative or expressive individuals
            3: 'Social',           # Example: People-oriented or helping individuals
            4: 'Enterprising',     # Example: Leadership or entrepreneurial individuals
            5: 'Conventional'      # Example: Structured or detail-oriented individuals
        }

        # Loop through each open-ended response
        for i, response in enumerate(responses):
            # Classify personality based on the response
            personality_class, _ = classify_personality(response)
            
            # Map the personality class to a RIASEC type
            riasec_type = mbti_to_riasec_mapping.get(personality_class, 'Realistic')  # Default to 'Realistic'
            
            # Increment the score for the corresponding RIASEC type
            riasec_scores[riasec_type] += 1

        # Sort RIASEC types by their scores in descending order and select the top two
        top_riasec_types = sorted(riasec_scores, key=riasec_scores.get, reverse=True)[:2]

        # Create a prompt for job recommendations based on the top RIASEC types
        prompt = (f"I have a person with RIASEC types {top_riasec_types}. "
                  "Can you suggest a few career options and explain why they might be suitable?")

        # Tokenize the prompt and generate job suggestions using the model
        input_ids = mbti_tokenizer(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            model_outputs = mbti_model.generate(input_ids, max_length=150)
        
        # Decode the model output to get the job recommendations as text
        job_recommendations = mbti_tokenizer.decode(model_outputs[0], skip_special_tokens=True)

        return top_riasec_types, job_recommendations

    except Exception as error:
        logging.error(f"Error processing responses: {error}")
        raise

# Example usage with a list of open-ended responses
sample_responses = [
    "I enjoy solving problems that involve technical work.",
    "I love working with others and helping them succeed.",
    "I often express myself through creative writing and art.",
    "I like organizing data and following structured processes.",
    "I'm good at leading teams and managing projects."
]

# Main execution block to handle command-line arguments
try:
    if len(sys.argv) != 4:
        print("Usage: python finetune_llama.py <param1> <param2> <param3>")
        sys.exit(1)

    # Retrieve command-line argument for open-ended responses (param1)
    param1 = sys.argv[1]  # Example: could be a JSON string or a set of text responses
    
    # Process the responses and get job suggestions
    process_open_ended_responses(param1)

except Exception as error:
    logging.error(f"Error during job suggestion process: {error}")
    raise
