from .vlm import make_llm_call_gemini, make_llm_call_vertex
from ..models.types import ElementType, Element, VLMConfig
from .file_system import FileSystem
from .element_types import (
    get_visual_elements_as_str, 
    get_non_visual_elements_as_str, 
    get_element_description_block, 
    default_element_types,
    get_num_visual_elements,
    get_num_non_visual_elements,
)
from pdf2image import convert_from_path, pdfinfo_from_path
import json
import time
import concurrent.futures

"""
pip install pdf2image
brew install poppler
"""

SYSTEM_MESSAGE = """
You are a PDF -> MD file parser. Your task is to analyze the provided PDF page (provided as an image) and return a structured JSON response containing all of the elements on the page. Each element must be represented using Markdown formatting.

There are two categories of elements you need to identify: text elements and visual elements. Text elements are those that can be accurately represented using plain text. Visual elements are those that need to be represented as images to fully capture their content. For text elements, you must provide the exact text content. For visual elements, you must provide a detailed description of the content.

There are {num_visual_elements} types of visual elements: {visual_elements_as_str}.
There are {num_non_visual_elements} types of text elements: {non_visual_elements_as_str}.

Every element on the page should be classified as one of these types. There should be no overlap between elements. You should use the smallest number of elements possible while still accurately representing and categorizing the content on the page. For example, if the page contains a couple paragraphs of text, followed by a large figure, followed by a few more paragraphs of text, you should use three elements: NarrativeText, Figure, and NarrativeText. With that said, you should never combine two different types of elements into a single element.

Here are detailed descriptions of the element types you can use:
{element_description_block}

For visual elements ({visual_elements_as_str}), you must provide a detailed description of the element in the "content" field. Also, please provide the description in the same language as the document. For example, if the document is in English, please provide the detailed description of the element in English. If the dominant language in the page is French, Please provide the description in French, etc. Do not just transcribe the actual text contained in the element, but make sure to do so when its possible along with the description of the element. For textual elements ({non_visual_elements_as_str}), you must provide the exact text content of the element. 

Output format
- Your output should be an ordered (from top to bottom) list of elements on the page, where each element is a dictionary with the following keys:
    - type: str - the type of the element
    - content: str - the content of the element. For visual elements, this should be a detailed description of the visual content, rather than a transcription of the actual text contained in the element. You can use Markdown formatting for text content.

Complex and multi-part figures or images should be represented as a single element. For example, if a figure consists of a main chart and a smaller inset chart, these should be described together in a single Figure element. If there are two separate graphs side by side, these should be represented as a single Figure element with a bounding box that encompasses both graphs. DO NOT create separate elements for each part of a complex figure or image.
"""

response_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
            },
            "content": {
                "type": "string",
            },
        },
        "required": ["type", "content"],
    },
}

def pdf_to_images(pdf_path: str, kb_id: str, doc_id: str, file_system: FileSystem, dpi=200) -> list[str]:
    """
    Convert a PDF to images and save them to a folder. Uses pdf2image (which relies on poppler).

    Inputs:
    - pdf_path: str - the path to the PDF file.
    - page_images_path: str - the path to the folder where the images will be saved.

    Returns:
    - image_file_paths: list[str] - a list of the paths to the saved images.
    """
    
    # Create the folder
    file_system.create_directory(kb_id, doc_id)
    print("Converting PDF to images")
    info = pdfinfo_from_path(pdf_path, userpw=None, poppler_path=None)
    maxPages = info["Pages"]
    batch_size = 10
    image_file_paths = []
    for batch_pages in range(1, maxPages+1, batch_size) : 
        images = convert_from_path(pdf_path, dpi=200, first_page=batch_pages, last_page = min(batch_pages+batch_size-1,maxPages))
        print(f"{batch_size+batch_pages} Converted")
        for i, image in enumerate(images):
            #image_file_path = os.path.join(page_images_path, f'page_{i+1}.png')
            file_system.save_image(kb_id, doc_id, f'page_{i+batch_pages}.png', image)
            #image.save(image_file_path, 'PNG')
            image_file_path = f'/{kb_id}/{doc_id}/page_{i+batch_pages}.png'
            image_file_paths.append(image_file_path)
        print(f"{batch_size+batch_pages} saved")

    # # Convert PDF to images
    # images = convert_from_path(pdf_path, dpi=dpi, thread_count=100)
    # print("Images converted")
    # # Save each image
    # image_file_paths = []
    # for i, image in enumerate(images):
    #     #image_file_path = os.path.join(page_images_path, f'page_{i+1}.png')
    #     file_system.save_image(kb_id, doc_id, f'page_{i+1}.png', image)
    #     #image.save(image_file_path, 'PNG')
    #     image_file_path = f'/{kb_id}/{doc_id}/page_{i+1}.png'
    #     image_file_paths.append(image_file_path)

    print(f"Converted {maxPages} pages to images")
    return image_file_paths

def parse_page(kb_id: str, doc_id: str, file_system: FileSystem, page_number: int, vlm_config: VLMConfig, element_types: list[ElementType]) -> list[Element]:
    """
    Given an image of a page, use LLM to extract the content of the page.

    Inputs:
    - page_image_path: str, path to the image of the page
    - vlm_config: dict, configuration for the VLM
    - element_types: list[ElementType], list of element types that the VLM can output
    
    Outputs:
    - page_content: list of Elements
    """

    # use default vlm_provider and model if not provided
    if "provider" not in vlm_config:
        vlm_config["provider"] = "gemini"
    if "model" not in vlm_config:
        if vlm_config["provider"] == "gemini":
            vlm_config["model"] = "gemini-2.0-flash"
        else:
            raise ValueError("Non-default VLM provider specified without specifying model")

    # format system message
    system_message = SYSTEM_MESSAGE.format(
        num_visual_elements=get_num_visual_elements(element_types),
        num_non_visual_elements=get_num_non_visual_elements(element_types),
        visual_elements_as_str=get_visual_elements_as_str(element_types),
        non_visual_elements_as_str=get_non_visual_elements_as_str(element_types),
        element_description_block=get_element_description_block(element_types)
    )

    page_image_path = file_system.get_files(kb_id, doc_id, page_number, page_number)[0]

    if vlm_config["provider"] == "vertex_ai":
        try:
            # Get temperature from vlm_config or use default
            # NOTE: it's very important to use a non-zero temperature here
            # Using a temp of 0 causes frequent degenerative output that can't be fixed by retrying
            temperature = vlm_config.get("temperature", 0.7) 
            llm_output = make_llm_call_vertex(
                image_path=page_image_path, 
                system_message=system_message, 
                model=vlm_config["model"], 
                project_id=vlm_config["project_id"], 
                location=vlm_config["location"],
                response_schema=response_schema,
                max_tokens=4000,
                temperature=temperature
            )
        except Exception as e:
            if "429 Online prediction request quota exceeded" in str(e):
                print (f"Rate limit exceeded in make_llm_call_vertex: {e}")
                return 429
            else:
                print (f"Error in make_llm_call_gemini: {e}")
                error_data = {
                    "error": f"Error in make_llm_call_gemini: {e}",
                    "function": "parse_page",
                }
                try:
                    file_system.log_error(kb_id, doc_id, error_data)
                except:
                    print ("Failed to log error")
                finally:
                    return 429
                
    elif vlm_config["provider"] == "gemini":
        try:
            llm_output = make_llm_call_gemini(
                image_path=page_image_path, 
                system_message=system_message, 
                model=vlm_config["model"],
                response_schema=response_schema,
                max_tokens=4000
            )
        except Exception as e:
            if "429 Online prediction request quota exceeded" in str(e):
                print (f"Error in make_llm_call_gemini: {e}")
                return 429
            else:
                print (f"Error in make_llm_call_gemini: {e}")
                error_data = {
                    "error": f"Error in make_llm_call_gemini: {e}",
                    "function": "parse_page",
                }
                try:
                    file_system.log_error(kb_id, doc_id, error_data)
                except:
                    print ("Failed to log error")
                finally:
                    llm_output = json.dumps([{
                        "type": "text",
                        "content": "Unable to process page"
                    }])
                    
    else:
        raise ValueError("Invalid provider specified in the VLM config. Only 'vertex_ai' and 'gemini' are supported for now.")
    
    try:
        page_content = json.loads(llm_output)
    except Exception as e:
        print(f"Error for {page_image_path}: {e}")
        error_data = {
            "error": f"Error parsing JSON for {page_image_path}: {e}",
            "function": "parse_page",
        }
        try:
            file_system.log_error(kb_id, doc_id, error_data)
        except:
            print ("Failed to log error")
        page_content = []

    # add page number to each element
    for element in page_content:
        element["page_number"] = page_number

    return page_content

def parse_file(pdf_path: str, kb_id: str, doc_id: str, vlm_config: VLMConfig, file_system: FileSystem) -> list[Element]:
    """
    Given a PDF file, extract the content of each page using a VLM model.
    
    Inputs
    - pdf_path: str, path to the PDF file - can be an empty string if images_already_exist is True
    - kb_id: str, knowledge base ID
    - doc_id: str, document ID
    - vlm_config: dict, configuration for the VLM model. For Vertex this should include project_id and location.
    - file_system: FileSystem, object for interacting with the file system where the images are stored
    
    Outputs
    - all_page_content: list of Elements

    Saves
    - images of each page of the PDF (if images_already_exist is False)
    - JSON files of the content of each page
    """
    images_already_exist = vlm_config.get("images_already_exist", False)
    if images_already_exist:
        image_file_paths = file_system.get_all_png_files(kb_id, doc_id)
    else:
        image_file_paths = pdf_to_images(pdf_path, kb_id, doc_id, file_system)
    
    all_page_content_dict = {}

    element_types = vlm_config.get("element_types", default_element_types)
    if len(element_types) == 0:
        element_types = default_element_types

    def process_page(page_number):
        tries = 0
        while tries < 20:
            content = parse_page(
                kb_id=kb_id,
                doc_id=doc_id,
                file_system=file_system,
                page_number=page_number,
                vlm_config=vlm_config, 
                element_types=element_types
            )
            if content == 429:
                print(f"Rate limit exceeded. Sleeping for 10 seconds before retrying...")
                time.sleep(10)
                tries += 1
                continue
            # Check if the content is empty - a signal that JSON parsing failed
            if isinstance(content, list) and len(content) == 0:
                # This suggests we had a JSON parsing error
                print(f"Empty content returned, likely due to JSON parsing error. Retrying... (retry_attempt = {tries+1})")
                tries += 1
                continue
            else:
                return page_number, content
        return page_number, [{"type": "NarrativeText", "content": "Failed to process page after multiple attempts", "page_number": page_number}]

    # Use ThreadPoolExecutor to process pages in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_page, i + 1): i for i in range(len(image_file_paths))}
        for future in concurrent.futures.as_completed(futures):
            page_content = future.result()
            # Add the page content to the dictionary, keyed on the page number
            page_number, page_content = future.result()
            all_page_content_dict[page_number] = page_content

    all_page_content = []
    for key in sorted(all_page_content_dict.keys()):
        all_page_content.extend(all_page_content_dict[key])

    # Save the extracted content to a JSON file
    file_system.save_json(kb_id, doc_id, 'elements.json', all_page_content)

    return all_page_content

def elements_to_markdown(elements: list[Element]) -> str:
    """
    Given a list of elements extracted from a PDF, convert them to a markdown string.
    
    Inputs
    - elements: list of dictionaries, each containing information about an element on a page
    
    Outputs
    - markdown_string: str, a markdown string representing the elements
    """
    markdown_string = ""
    for element in elements:
        markdown_string += f"{element['content']}\n\n"

    return markdown_string