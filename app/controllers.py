from app.models import df_llm

def process_prompt(prompt):
    try:
        response = df_llm.chat(prompt)
        # After generating charts, check and upload new images
        from app.models import check_for_new_images, watch_directory
        print("Checking for new images to upload after generating charts...")
        latest_image_url = check_for_new_images(watch_directory)
        return {
            "response": response,
            "latest_image_url": latest_image_url
        }
    except Exception as e:
        return str(e)
