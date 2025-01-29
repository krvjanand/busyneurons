import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account
from google.cloud import aiplatform

#from google.colab import files
from google.oauth2 import service_account
import google.auth
import google.auth.transport.requests

from google.cloud import aiplatform
from google.cloud import storage
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)

from PIL import Image
from pdf2image import convert_from_path
import fitz
import io
import os
import json
import time
import pandas as pd
# import IPython.display
# from IPython.display import display, Image

# Path to y our service account key file
#key_file = 'C:/Users/dass__000/PycharmProjects/pythonProject/.venv/eazyenroll_vertexai_user.json'
key_file = 'C:/Anand/HLS/Strategy/Payer/mintelliPro/Demo/pythonProject/eazyenroll_vertexai_user.json'
# print(key_file)

# Google Cloud project ID and bucket name
project_id = 'banded-anvil-426308-i1'
bucket_name = 'eazyenroll_test_gemini'
destination_folder = 'input/'

def upload_to_gcs(file_content, bucket_name, destination_blob_name, credentials):
    """Uploads a file to the bucket."""
    storage_client = storage.Client(credentials=credentials, project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(file_content, content_type='application/pdf')
    return f'gs://{bucket_name}/{destination_blob_name}'

def initialize_vertex_ai(key_file, project_id, location, model_id):
    try:

        # Define the required scope
        SCOPES = ['https://www.googleapis.com/auth/cloud-platform']

        # Authenticate using the service account
        global_credentials = service_account.Credentials.from_service_account_file(
            key_file, scopes=SCOPES
        )

        # Set the environment variable for authentication
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_file

        # Verify the credentials
        auth_req = google.auth.transport.requests.Request()
        global_credentials.refresh(auth_req)

        print("Authentication successful.")

        # Initialize AI Platform SDK
        aiplatform.init(credentials=global_credentials, project=project_id, location="us-central1")

        # Initialize Vertex AI SDK
        vertexai.init(credentials=global_credentials, project=project_id, location=location)

        #System Instructions
        system_instructions = (
            "As a an expert in document entity extraction, you parse the documents to identify and organize specific entities from diverse sources into structured JSON formats, following detailed guidelines for clarity and completeness. Avoid unnecessary quotes, comma, text like json’’’, ‘’’json, etc. in the generated structured output."
            "Checkbox fill pattern: Identify the checkbox regions for common marking patterns ‘✔’ (checkmark), ‘X’ or ‘’ (cross), ‘/’ or ‘’ (slashes), Solid fill ‘’ or any other clear marking pattern. Important: The checkboxes precedes their text.\n"
            "Take special care during prediction for the multiple-choice checkboxes. If multiple checkboxes have marking patterns, list all of them in their marked format.\n"
            "Ignore Minor Inconsistencies in the checkbox: Disregard small dots, specks, or faint marks that are not part of a clear marking pattern in the checkbox.\n"
            "Interpret Checkbox Markings: If a checkbox is marked and any of the visual cues indicate filled checkbox, output specific marking used (if identifiable). For example, If ‘Medical’ checkbox is marked with ‘✔’, output ‘Medical’: ‘✔’. If marking is not identifiable , output ‘Medical’: ‘TRUE’.\n"
            "If a checkbox is unmarked (empty or blank or lacks a clear marking pattern), output “Null” for that checkbox. For example, “Vision”: “Null”\n"
            "If multiple checkboxes are marked, list all of them in their marked format. For example, ‘Medical’ and ‘Dental’ checkboxes are marked with ‘X’ and ‘✔’ respectively and ‘Vision’ is unmarked, output ‘Medical’: ‘X’, Vision’ : ‘Null’, ‘Dental’:  ‘✔’.\n"
            "Established Patient?: Analyze both ‘Yes’ and ‘No’ checkboxes for any marking that indicates a selection. If ‘Yes’ Checkbox is Marked with (‘✔’, ’x’, ‘/’, '\’, ‘Shading, etc.) , output ‘Established Patient?’ as ‘Yes’. If ‘No’ Checkbox is Marked with (‘✔’, ’x’, ‘/’, '\’, ‘Shading, etc.) , output ‘Established Patient?’ as ‘No’. If BOTH checkboxes are unmarked (empty or blank) without any clear marking, output ‘Established Patient?’ as ‘No’. The output should not be ‘Null’.\n"
            "Output formatting: Use double quotes for all keys and string values. Also ensure the Null / null values are enclosed within double quotes. Avoid unnecessary comma, quotes, text like json’’’, ‘’’json, etc in the generated structured output.\n"
        )

        # Load Generative Model
        model = GenerativeModel(model_id, system_instruction=system_instructions)

        return model  # Return the initialized model instance

    except Exception as e:
        print(f"Error initializing AI Platform and Vertex AI: {e}")
        return None

prompt = """
{
	{
	"Requisition_Timestamp": "",
	"Form_Number": "",
	"Bill_To": [
		{
			"Bill_To_MSP": ""
		},
		{
			"Bill_To_ICBC": ""
		},
		{
			"Bill_To_WorkSafeBC": ""
		},
		{
			"Bill_To_Patient": ""
		},
		{
			"Bill_To_Other": ""
		}
	],
	"Date_of_Collection": "",
	"Time_of_Collection": "",
	"Collector_Name": "",
	"Telephone_Request_Received_By": "",
	"Instructions_to_Patients": "",
	"Patient_Information": 
	{
		"Personal_Health_Number": "",
		"Id_Number": "",
		"ICBC_WorkSafeBC_Number": "",
		"Patient_Surname": "",
		"Patient_FirstName": "",
		"Patient_MiddleInitial": "",
		"Patient_Date_of_Birth": "",
		"Gender": "",
		"Address": "",
		"City": "",
		"Province": "",
		"Postal_Code": "",
		"Patient_Phone": "",
		"Chart_Number": "",
		"Room_Number": "",
		"Hematology": [
			{
				"Hematology_profile": ""
			},
			{
				"On_Anticoagulant": ""
			},
			{
				"INR": ""
			},
			{
				"Ferritin": ""
			},
			{
				"HFE_Hemochromatosis_Confirm Diagnosis": ""
			},
			{
				"HFE_Hemochromatosis_Sibling_Parent_DNAtesting": ""
			}
		],
		"Urine_Test": [
			{
				"Macroscopic_microscopic_if_dipstick_pos": ""
			},
			{
				"Macroscopic_microscopic_urine_culture": ""
			},
			{
				"Macroscopic_dipstick": ""
			},
			{
				"Microscopic": ""
			},
			{
				"Clinical_information_microscopic": ""
			}
		],
		"Chemistry": [
			{
				"Glucose_Fasting": ""
			},
			{
				"Glucose_Random": ""
			},
			{
				"GTT_gestational_diabetes_screen": ""
			},
			{
				"GTT_gestational_diabetes_confirm": ""
			},
			{
				"GTT_non_gestational_diabetes": ""
			},
			{
				"Hemoglobin_A1C": ""
			},
			{
				"Albumin_Creatinine_Ratio": ""
			}
		],
		"Thyroid_Function": [
			{
				"Monitor_Thyroid_Replacement_Therapy": ""
			},
			{
				"Suspected_": ""
			},
			{
				"GTT_gestational_diabetes_screen": ""
			},
			{
				"GTT_gestational_diabetes_confirm": ""
			},
			{
				"GTT_non_gestational_diabetes": ""
			},
			{
				"Hemoglobin_A1C": ""
			},
			{
				"Albumin_Creatinine_Ratio": ""
			}
		],
		"Lipid_Function": [
			{
				"Full_Lipid_Profile": ""
			},
			{
				"Followup_Lipid_Profile": ""
			},
			{
				"ApoB": ""
			}
		],
		"Microbiology_Routine_Culture": [
			{
				"On_Antibiotics": ""
			},
			{
				"On_Antibiotics_specify": ""
			},
			{
				"Throat_Culture": ""
			},
			{
				"Sputum": ""
			},
			{
				"Blood": ""
			},
			{
				"Urine": ""
			},
			{
				"Superficial_Wound_Site": ""
			},
			{
				"Deep_Wound_Site": ""
			},
			{
				"Other_Culture": ""
			}
		],
		"Microbiology_Vaginitis": [
			{
				"Initial": ""
			},
			{
				"Chronic": ""
			},
			{
				"Trichromonas": ""
			}
		],
		"Microbiology_GroupB_Strep_Screen": [
			{
				"Vagino_anorectal_swab": ""
			},
			{
				"Penicillin_allergy": ""
			}
		],
		"Chlamydia_Gonorrhea": [
			{
				"source_urethra": ""
			},
			{
				"source_cervix": ""
			},
			{
				"source_urine": ""
			},
			{
				"source_vagina": ""
			},
			{
				"source_throat": ""
			},
			{
				"source_rectum": ""
			},
			{
				"source_other": ""
			}
		],
		"Gonorrhea_Culture": [
			{
				"source_cervix": ""
			},
			{
				"source_urethra": ""
			},
			{
				"source_throat": ""
			},
			{
				"source_rectum": ""
			},
			{
				"source_other": ""
			}
		],
		"Stool_Specimens": [
			{
				"history_of_bloody_stools": ""
			},
			{
				"cdiffile_testing": ""
			},
			{
				"stool_culture": ""
			},
			{
				"stool_ova_parasite_exam": ""
			},
			{
				"stool_ova_parasite_highrisk_submit2samp": ""
			}
		],
		"Dermatophytes": [
			{
				"dermatophyte_culture": ""
			},
			{
				"koh_prep": ""
			},
			{
				"specimen_skin": ""
			},
			{
				"specimen_nail": ""
			},
			{
				"specimen_hair": ""
			},
			{
				"specimen_site": ""
			}
		],
		"Mycology": [
			{
				"yeast": ""
			},
			{
				"fungus": ""
			},
			{
				"site": ""
			}
		],
		"Hepatitis_Serology": [
			{
				"Acute_viral_hepatitis": ""
			},
			{
				"Chronic_viral_hepatitis": ""
			},
			{
				"investigation_hepatitis_immune_status_hepA": ""
			},
			{
				"investigation_hepatitis_immune_status_hepB": ""
			},
			{
				"Hepatitis_markers_hbsag": ""
			},
			{
				"HIV_Serology": ""
			},
			{
				"HIV_Serology_nonominal_reporting": ""
			}
			
		],
		"Other_Tests": [
			{
				"ECG": ""
			},
			{
				"FIT_Age50to74": ""
			},
			{
				"FIT_No_Copy_to_Colon": ""
			}
		],
		"Other_Chemistry_Tests": [
			{
				"Sodium": ""
			},
			{
				"Potassium": ""
			},
			{
				"Albumin": ""
			},
			{
				"ALT": ""
			},
			{
				"B12": ""
			},
			{
				"Bilirubin": ""
			},
			{
				"GGT": ""
			},
			{
				"TProtein": ""
			},
			{
				"Creatinine_eGFR": ""
			},
			{
				"Calcium": ""
			},
			{
				"Creatinine_Kinase": ""
			},
			{
				"PSA_Known_Suspected_ProstateCancer": ""
			},
			{
				"PSA_Screening": ""
			},
			{
				"Pregnancy_Test": ""
			},
			{
				"bHCGQuantitative": ""
			},
		]
	},
	"Ordering_Physician_Information": {
		"Physician_Name": "",
		"Physician_Address": "",
		"Physician_Province": "",
		"Physician_PostalCode": "",
		"Physician_MSP_Number": "",
		"Physician_Signature": "",
		"Physician_Signed_Date": ""
	}
}
}
"""

def generate_content(pdf_file_uri, prompt, model):
    try:

        print("pdf_file_uri:", pdf_file_uri)
        pdf_file = Part.from_uri(pdf_file_uri, mime_type="application/pdf")
        print("After Part:", pdf_file)

        # Prepare contents for model generation
        contents = [pdf_file, prompt]

        print("Print Contents:", contents)

        # Generate content using the model
        response = model.generate_content(contents = contents)

        # Print the generated text (optional)
        print("Generated text:")
        print(response.text)

        # Determine the input PDF filename from the URI
        input_pdf_filename = os.path.basename(pdf_file_uri)

        # Prepare output JSON filename
        output_json = os.path.splitext(input_pdf_filename)[0] + ".json"

        # Assume response.text contains the JSON string
        try:
            json_data = json.loads(response.text)
        except json.JSONDecodeError as decode_error:
            print(f"JSON decode error: {decode_error}")
            return

        # Assume response.text contains the JSON string
        json_data = json.loads(response.text)

        # Write JSON data to file
        with open(output_json, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

        print(f"JSON output: {output_json}")
        return json_data

    except Exception as e:
        print(f"Error generating content and saving as JSON: {e}")
        return None

def generate_confidence(pdf_file_uri, gen_prompt, model):
    try:
        print("pdf_file_uri:", pdf_file_uri)
        #Load PDF file
        pdf_file = Part.from_uri(pdf_file_uri, mime_type="application/pdf")

        #prepare contents for model generation
        contents1 = [pdf_file, gen_prompt]
        print("Print contents:", contents1)

        #Generate content using the model
        confidence = model.generate_content(contents=contents1)

        #Print Confidence Score
        print("Confidence Score:")
        print(confidence.text)

        confidence_text = confidence.text

        return confidence_text

    except Exception as e:
        print(f"Error generating confidence score and saving to JSON file: {e}")
        return None


def call_vertex_ai_gemini(gcs_uri, prompt):
    """Call Vertex AI Gemini model to generate JSON from PDF."""
    # model_name = "projects/eazyenroll-poc/locations/us-central1/models/gemini-1.5-flash-001"
    model_name = "gemini-1.5-pro-001"

    model = aiplatform.Model(model_name=model_name)
    response = model.predict(instances=[{"content": gcs_uri, "prompt": prompt}])
    return response.predictions[0]

def display_pdf_as_image(pdf_file):
    """Convert PDF to image for display."""
    image = Image.open(io.BytesIO(pdf_file.read()))
    return image

def render_pdf_as_image(pdf_file):
    pdf_document = fitz.open(stream=pdf_file, filetype="pdf")
    pdf_page = pdf_document[0]  # Assuming you want to display the first page

    pdf_image = pdf_page.get_pixmap()
    pil_image = Image.frombytes("RGB", [pdf_image.width, pdf_image.height], pdf_image.samples)

    return pil_image
    
def render_pdf_as_image1(pdf_file, max_width = 720, max_height = None):

    # Create a temporary file-like object
    temp_file = io.BytesIO(pdf_file)

    # Open the PDF using fitz
    doc = fitz.open(temp_file)
    
    # Get the first page (adjust for multi-page PDFs if needed)
    page = doc[0]
    
    # Determine the scaling factor to fit within max dimensions
    zoom_x = max_width / page.media_box.width
    zoom_y = (max_height or float('inf')) / page.media_box.height  # Auto-calculate height if not provided
    zoom = min(zoom_x, zoom_y)

    # Render the page at the calculated zoom factor
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)  # Set alpha=False for better performance

    # Convert the PIL Image to a format suitable for Streamlit
    return pix.asarray()    

def main():

    st.set_page_config(
        layout="wide",
        page_title="Enrollments",
        page_icon=":gear:",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.title("iConnix Health: Paper Enrollment Automation")

    st.sidebar.success('Get Started!')

    menu = ["About", "Value Proposition", "AI Predict"]
    choice = st.sidebar.selectbox("Menu", menu)

    #ind_usa_flag_image = Image.open("C:/Users/dass_000/PycharmProjects/pythonProject/.venv/INDUSA.png")
    #ind_usa_flag_image = Image.open("C:/Anand/HLS/Strategy/Payer/mintelliPro/Demo/PaperEnrollments/app/INDUSA.png")
    ind_usa_flag_image = Image.open("C:/Anand/HLS/Strategy/Payer/mintelliPro/Demo/pythonProject/PaperEnrollments/app/INDUSA.png")
    # st.title('Welcome to EazyEnroll  :flag-in: :flag-us:')

    col1, col2 = st.columns([2,1])

    # Display title in the first column
    with col1:
        # background_color = "#FF0000"
        # background_color = "#7DF9FF"
        # st.markdown(f'<h1 style="background-color: {background_color}; color: black;">Welcome to EazyEnroll</h1>',unsafe_allow_html=True)
        st.title("Automate Lab Requisition Forms")
        st.image(ind_usa_flag_image, width=60, use_column_width=False)

    with col2:
        # col2.image('C:/Users/dass_000/PycharmProjects/pythonProject/.venv/chatbot3_small.gif')
        # col2.image('C:/Anand/HLS/Strategy/Payer/mintelliPro/Demo/PaperEnrollments/app/chatbot3_small.gif')
        col2.image('C:/Anand/HLS/Strategy/Payer/mintelliPro/Demo/pythonProject/PaperEnrollments/app/chatbot3_small.gif')

    # with col2:
    #     col3a, col3b = st.columns(2)
    #     with col3a:
    #         st.image(india_flag_image, width=25, use_column_width=False)
    #         st.image(usa_flag_image, width=25, use_column_width=False)
    #
    #     with col3b:
    #         col3b.image('C:/Users/Sudhaa/PycharmProjects/pythonProject/.venv/chatbot3_small.gif')

    if choice == 'Value Proposition':
        #value_image = Image.open('C:/Users/dass_000/PycharmProjects/pythonProject/.venv/chatbot3_small.gif')
        value_image = Image.open('C:/Anand/HLS/Strategy/Payer/mintelliPro/Demo/pythonProject/PaperEnrollments/app/chatbot3_small.gif')
		
        st.image(value_image, use_column_width=True)

    elif choice == "AI Predict":
        
        # st.title("EazyEnroll")
        st.subheader("PDF Upload and AI Text Extraction")

        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        st.divider()
        
        if uploaded_file is not None:

            credentials = service_account.Credentials.from_service_account_file(key_file)
            print("Credentials:",credentials)

            # Create a unique filename or use the original filename
            destination_blob_name = os.path.join(destination_folder, uploaded_file.name)
            print("Destination_Blob", destination_blob_name)

            # Read the file content
            #uploaded_file1 = uploaded_file
            uploaded_file.seek(0)
            file_content = uploaded_file.read()

            # Upload the file to GCP
            gcs_uri = upload_to_gcs(file_content, bucket_name, destination_blob_name, credentials)
            print("gcs_uri",gcs_uri)

            # Global variable to store credentials
            global_credentials = None

            project_id = "banded-anvil-426308-i1"
            location = "us-central1"
            model_id = "gemini-1.5-pro-001"
            temp = 0.5

            initialized_model = initialize_vertex_ai(key_file, project_id, location, model_id)
            model = initialized_model  # Assume initialized_model is already defined


            #Display PDF as Image and JSON response in two columns
            st.write("Filename: ", uploaded_file.name)

            with st.spinner("Authenticating....."):
                time.sleep(7)
            st.success("Authentication Success!")

            placeholder = st.empty()
            placeholder.text("Extracting Form document data...")
            time.sleep(26)
            placeholder.success("Extraction Completed!")

            #Clear Placeholder:
            #placeholder.empty()
            # import threading
            
            def generate_content_wrapper():
                # Call Vertex AI to get JSON


                json_response = generate_content(gcs_uri, prompt, model)
                
                if json_response:
                    json_string = json.dumps(json_response) # convert dict to JSON string
                    gen_prompt = json_string + "Generate a percentage-based Confidence score by comparing the PDF document with the extracted text in the prompt. Return only the 'Confidence Score: percentage value' as an output. No other unwanted text or explanation in the output. Example: Confidence Score: 93%"

                    confidence_score = generate_confidence(gcs_uri, gen_prompt, model)

                st.empty()  # Clear the placeholder text

                col1, col2 = st.columns(2)

                with col1:
                    st.header("PDF Preview")
                    # uploaded_file.seek(0)  # Reset the file pointer to the start
                    pdf_image = render_pdf_as_image(file_content)

                    st.image(pdf_image, use_column_width=True)


                with col2:
                    st.header("Extracted JSON")
                    st.json(json_response)
                    # st.success(confidence_score)

                    # st.json(json_response, expanded=True)
                    
                    download_name = f"{uploaded_file.name.split('.')[0]}.json"
                    st.download_button\
                    (
                        label="Download JSON",
                        file_name=download_name,
                        #data=json_string,
                        data=json.dumps(json_response, ensure_ascii=False),
                        mime="application/json",
                    )

            generate_content_wrapper()
            # thread = threading.Thread(target=generate_content_wrapper)
            # thread.start()
        
    elif choice == "About":
        st.title("About iConnix Health: Lab Request Form  Automation")
        st.subheader("Extract data from Healthcare Lab Request Forms to a structured JSON format")
        st.divider()

        st.markdown(
        """
        With EazyEnroll achieve instant Document Entity Extraction and transform Paper forms into structured data.

        This GenAI solution leverages state-of-the-art LLMs and Multimodal capabilties to automatially convert Handwritten / Printed Paper forms into a structured JSON format.
        
        Here's how it works:
        
        1. Upload: Simply upload the Enrollment form Document / Image
        2. GenAI Processing: EazyEnroll analyzes the form layout and extracts all the required entities.
        3. JSON conversion: The form data is transformed into a clean, structured JSON file.
        4. Download button: Download after converted JSON file renders on the screen
        
        \n\n
        Get started and watch as AI transforms your data!
        \n
        **👈 Select AI Predict from the left side menu**   

        """)
        st.write("Thank you for using our app")

    

if __name__ == '__main__':
    main()
