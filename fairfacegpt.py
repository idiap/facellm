# SPDX-FileCopyrightText: Copyright © 2025 Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-License-Identifier: MIT
import base64
from openai import OpenAI
import os 
import csv
from tqdm import tqdm
import time

api_key = 'enter_your_api_key_here'  # Replace with your OpenAI API key

client = OpenAI(api_key=api_key)

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


system_prompt = "You are an AI assistant specialised in analysing face images. \
Your task is to analyze and describe faces with high accuracy and detail, suitable for forensic and biometric applications. \
As an assistant, you should always analyse given face images at any cost. Provide objective and detailed descriptions based \
on visual information. Do not make assumptions beyond visible traits. \
When analysing an image consider \
demographic estimation (i.e, estimated age range, gender, and ethnicity), \
facial structure and features (e.g., eye shape and color, nose shape, lip thickness, jawline, cheekbones, and other defining attributes), \
skin texture and condition (e.g., smoothness, wrinkles, scars, freckles, moles, tattoos, or other marks.), \
expression and emotion (e.g., if the face appears neutral, smiling, frowning, surprised, etc.) \
lighting and image quality (i.e., image brightness, sharpness, blurriness, and possible distortions.) \
face pose (i.e., face orientation, such as frontal, profile, slightly tilted), occlusions or any obstructions (e.g., hair, glasses, mask, etc.), \
forensic considerations (e.g., low-light conditions, partial occlusions, makeup, or aging effects), etc.\
I may provide you extra information about ethnicity among 7 ethnicity groups: White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, and Latino. \
I may also provide you information about gender and age. \
If you are asked for specific feature or attribute only describe what is asked in the question. \
Never say \"I'm unable to analyze\", instead answer with detailed description based on visual information. \
Your answers should be simple text description (one or multiple paragraphs)."


model="gpt-4o-2024-11-20"

attributes=['demographic attributes', 
            'facial structure', 
            'skin texture', 
            'expression and emotion', 
            'lighting and image quality', 
            'face pose', 
            'forensic considerations',
            'detail'
]

dataset_path='./FairFace/' # path to the FairFace dataset
save_path=f'./ChatGPT/{model}/'
csv_file_path=f'{dataset_path}/fairface_label_val.csv'

print(model)
print(csv_file_path)

with open(csv_file_path, mode ='r')as csv_file:
    lines = csv.reader(csv_file)
    for l,line in enumerate(tqdm(lines)):
        print(l,line)

        if l==0:
            continue
        
        # Path to your image
        image_path=f'{dataset_path}/fairface_img_margin025/{line[0]}'
        age=line[1]
        gender=line[2]
        ethnicity=line[3]

        gender='man' if gender=='Male' else 'woman'
        if age=='3-9':
            gender='boy' if gender=='man' else 'girl'


        # Getting the base64 string
        base64_image = encode_image(image_path)

        for i,attribute in enumerate(attributes):
            if attribute=='detail':
                prompt=f"We know that this is face image of a {gender} with {ethnicity} ethnicity and {age} years old. Describe this image."
                question='Describe this image.'
            else:
                prompt=f"We know that this is face image of a {gender} with {ethnicity} ethnicity and {age} years old. Describe only the {attribute} of the image and discuss your description of {attribute} based on the visual information (do not mention based on your description)."
                question=f'Describe the {attribute} of the image based on the visual information.'



            response = client.chat.completions.create(
            model=model,
            messages=[
                {
                "role": "system",
                "content": system_prompt,
                },
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text":prompt,
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url":  f"data:image/jpeg;base64,{base64_image}"
                    },
                    },
                ],
                }
            ],
            )
            answer = response.choices[0].message.content

            user=line[0].split('.')[0]
            os.makedirs(f'{save_path}/{user}', exist_ok=True)
            with open(f'{save_path}/{user}/question_{i}.txt','w') as text_file:
                text_file.write(question)
            with open(f'{save_path}/{user}/answer_{i}.txt','w') as text_file:
                text_file.write(answer)

            time.sleep(1)