import streamlit as st
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
from openai import OpenAI
import openai
import requests
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

PARENT_DIR = Path(__file__).parent
MP3_DIR = PARENT_DIR / "mp3s"

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if OPENAI_API_KEY is None:
    raise ValueError("API_KEY environment variable is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

def create_seed_ideas(song_info, selected_fields, custom_aesthetic):
    """
    Creates seed ideas for generating cover art based on song information.

    Args:
        song_info (dict): A dictionary containing song information.
        selected_fields (list): A list of selected fields to include in the prompt.
        custom_aesthetic (str): Custom aesthetic elements to include in the prompt.

    Returns:
        list: A list of three seed ideas for cover art.
    """
    system_prompt = """You are a creative assistant who generates seed ideas for album cover art.
    Your task is to provide three distinct, brief ideas for album covers based on the given song information.
    Each idea should be a single sentence, focusing on a unique visual concept.
    Favour simple ideas over complex visuals. Do not include any words, text, letters or numbers to be displayed in the description.
    Do not include the name of the artist or the song in the art.
    Do not include any type of graphic imagery.
    Do not include any text or artist names in the ideas."""

    prompt_parts = []
    prompt_parts.append(f"Generate three distinct seed ideas for cover art for a song ")
    if 'Track' in selected_fields:
        prompt_parts.append(f"called '{song_info['Track']}' ")
    if 'Artist (lyrics)' in selected_fields:
        prompt_parts.append(f"by {song_info['Artist (lyrics)']}")
    if 'Genre' in selected_fields:
        prompt_parts.append(f". The genres are: {song_info['Genre']}")
    if 'Mood' in selected_fields:
        prompt_parts.append(f". The mood is: {song_info['Mood']}")
    if 'Lyrics' in selected_fields:
        prompt_parts.append(f"The lyrics are: {song_info['Lyrics']}...")
    if custom_aesthetic:
        prompt_parts.append(f"Include these aesthetic elements: {custom_aesthetic}")
    complete_prompt = ''.join(prompt_parts)

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": complete_prompt}
        ]
    )
    
    seed_ideas = completion.choices[0].message.content.split('\n')
    return [idea.strip() for idea in seed_ideas if idea.strip()]

def create_prompt(seed_idea, song_info, selected_fields, custom_aesthetic):
    """
    Creates a prompt for generating cover art based on a seed idea and song information.

    Args:
        seed_idea (str): A seed idea for the cover art.
        song_info (dict): A dictionary containing song information.
        selected_fields (list): A list of selected fields to include in the prompt.
        custom_aesthetic (str): Custom aesthetic elements to include in the prompt.

    Returns:
        str: The generated prompt for cover art.
    """
    system_prompt = """You are a helpful assistant who writes detailed descriptions of album covers. 
    Your responses must be at most 3 sentences with a precise description of the album art.
    Favour simple ideas over complex visuals. Describe the visuals exactly without suggestion. Do not start with 'the cover art...'. 
    Describe only what the cover art is. Include an aesthetic style in the description.
    Do not include any type of graphic imagery."""

    if seed_idea:
        prompt_parts = [f"Create a detailed description of cover art based on this idea: {seed_idea}. "]
    else:
        prompt_parts = ["Create a concise description of cover art for a song "]
    prompt_parts.append(f"The song genre is: {song_info['Genre']}. ")
    prompt_parts.append(f"The mood is: {song_info['Mood']}. ")
    prompt_parts.append(f"The lyrics are: {song_info['Lyrics']}... ")
    if custom_aesthetic:
        prompt_parts.append(f"Include these aesthetic elements: {custom_aesthetic}")
    prompt_parts.append("Do not include any words, text, letter or numbers to be depicted in the art. Do not include the name of the artist or the song in the art")
    complete_prompt = ''.join(prompt_parts)

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": complete_prompt}
        ]
    )
    image_prompt = completion.choices[0].message.content
    
    return image_prompt

def generate_image(prompt, style, song_info, selected_fields, custom_aesthetic, max_retries=3):
    """
    Generates an image based on the provided prompt using DALL-E 3.

    Args:
        prompt (str): The prompt to generate the image from.
        style (str): The style of the image, either "vivid" or "natural".
        max_retries (int): The maximum number of retries if DALL-E refuses to generate the image.

    Returns:
        tuple: A tuple containing the path to the generated image and the URL of the image.
    """
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                style=style,
                n=1,
            )
            image_url = response.data[0].url
            break
        except openai.error.InvalidRequestError as e:
            if e.code == "invalid_request":
                retry_count += 1
                prompt = create_prompt("", song_info, selected_fields, custom_aesthetic)
                if retry_count == max_retries:
                    raise e
            else:
                raise e

    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"generated_image_{timestamp}.png"
    os.makedirs("imgs", exist_ok=True)
    image_path = os.path.join("imgs", save_name)
    image.save(image_path)

    return image_path, image_url

@st.cache_data
def load_song_data():
    """
    Loads song data from a CSV file.

    Returns:
        pandas.DataFrame: The loaded song data.
    """
    return pd.read_csv(f"{PARENT_DIR}/db/song_info.csv")

def moderate(text):
    """
    Moderates the provided text using OpenAI's moderation API.

    Args:
        text (str): The text to moderate.

    Returns:
        bool: True if the text violates the moderation policy, False otherwise.
    """
    if not text:  # Guard API from empty strings
        return True
    r = client.moderations.create(input=f'Please create a detailed description of art depicting the following: {text}')
    content_violation = r.results[0].flagged
    return content_violation 
    
def main():
    """
    The main function that runs the Streamlit application for cover art generation.
    """
    st.title("Cover Art Generator Demo")

    song_data = load_song_data()

    song_options = [f"{row['Track']} - {row['Artist (lyrics)']}" for _, row in song_data.iterrows()]
    selected_song = st.selectbox("Select a song", song_options)

    st.subheader("Select information to use for image generation")
    use_artist = st.checkbox("Artist Name", value=True)
    use_song = st.checkbox("Song Name", value=True)
    use_genres = st.checkbox("Genres", value=True)
    use_mood = st.checkbox("Mood", value=True)
    use_lyrics = st.checkbox("Lyrics", value=True)

    custom_aesthetic = st.text_input("Enter any custom aesthetic elements (optional)")

    image_style = st.radio("Image Style", ("Natural", "Vivid"))
    image_style = image_style.lower()

    selected_index = song_options.index(selected_song)
    song_info = song_data.iloc[selected_index]

    selected_index = song_options.index(selected_song)
    song_info = song_data.iloc[selected_index]

    # Display music player
    mp3_file = song_info['mp3/wav']
    mp3_path = MP3_DIR / mp3_file
    if mp3_path.exists():
        st.subheader("Listen to the song")
        st.audio(str(mp3_path), format="audio/mpeg")
    else:
        st.warning(f"MP3 file not found: {mp3_file}")

    if st.button("Generate Cover Art"):
        selected_index = song_options.index(selected_song)
        song_info = song_data.iloc[selected_index]

        selected_fields = []
        if use_artist:
            selected_fields.append("Artist (lyrics)")
        if use_song:
            selected_fields.append("Track")
        if use_genres:
            selected_fields.append("Genre")
        if use_mood:
            selected_fields.append("Mood")
        if use_lyrics:
            selected_fields.append("Lyrics")

        flags = []
        if custom_aesthetic:
            if moderate(custom_aesthetic):
                flags.append('Custom Aesthetic')
        
        if use_lyrics:
            if moderate(song_info['Lyrics']):
                #flags.append('Lyrics')
                pass

        if flags:
            st.error(f"The following fields did not pass our moderation check: {', '.join(flags)}. Please ensure all content is appropriate.")
        else:
            seed_ideas = create_seed_ideas(song_info, selected_fields, custom_aesthetic)
            
            st.subheader("Generated Seed Ideas:")
            for i, idea in enumerate(seed_ideas, 1):
                st.info(f"Idea {i}: {idea}")

            images = []
            cols = st.columns(3)
            for i, (seed_idea, col) in enumerate(zip(seed_ideas, cols), 1):
                prompt = create_prompt(seed_idea, song_info, selected_fields, custom_aesthetic)
                with col:
                    st.text(f"Prompt {i}:")
                    st.info(prompt)

                    with st.spinner(f'Generating image {i}...'):
                        try:
                            image_path, image_url = generate_image(prompt, image_style, song_info, selected_fields, custom_aesthetic)
                            images.append((image_path, image_url))
                        except openai.error.InvalidRequestError:
                            st.warning(f"DALL-E refused to generate image {i} after multiple attempts. Skipping this image.")
                            continue
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_name = f"{song_info['Track'].replace(' ', '_')}_{timestamp}_{i}"
                    
                    os.makedirs("imgs", exist_ok=True)

                    with open(f"imgs/{save_name}_info.txt", "w") as f:
                        f.write(f"Song: {song_info['Track']}\n")
                        f.write(f"Artist: {song_info['Artist (lyrics)']}\n")
                        f.write(f"Genres: {song_info['Genre']}\n")
                        f.write(f"Mood: {song_info['Mood']}\n")
                        f.write(f"Lyrics: {song_info['Lyrics'][:100]}...\n")
                        f.write(f"Seed Idea: {seed_idea}\n")
                        f.write(f"Prompt: {prompt}\n")
                        f.write(f"Generated: {timestamp}\n")
                        f.write(f"Image URL: {image_url}\n")

                    st.image(image_path, caption=f"Cover art {i}", use_column_width=True, width=400)

            st.success("Three cover art variations generated successfully!")

if __name__ == "__main__":
    main()