from google.cloud import vision
from PIL import Image, ImageDraw

client = vision.ImageAnnotatorClient()

def detect_face(face_file, max_results=4):
    """Uses the Vision API to detect faces in the given file.

    Args:
        face_file: A file-like object containing an image with faces.

    Returns:
        An array of Face objects with information about the picture.
    """
    client = vision.ImageAnnotatorClient()

    content = face_file.read()
    image = vision.Image(content=content)

    return client.face_detection(image=image, max_results=max_results).face_annotations

def highlight_faces(image, faces, output_filename):
    """Draws a polygon around the faces, then saves to output_filename.

    Args:
      image: a file containing the image with the faces.
      faces: a list of faces found in the file. This should be in the format
          returned by the Vision API.
      output_filename: the name of the image file to be created, where the
          faces have polygons drawn around them.
    """
    im = Image.open(image)
    draw = ImageDraw.Draw(im)
    # Sepecify the font-family and the font-size
    for i, face in enumerate(faces):
        box = [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
        draw.line(box + [box[0]], width=5, fill="#00ff00")
        # Place the confidence value/score of the detected faces above the
        # detection box in the output image
        draw.text(
            (
                (face.bounding_poly.vertices)[0].x,
                (face.bounding_poly.vertices)[0].y - 30,
            ),
            str(format(face.detection_confidence, ".3f")) + "%",
            fill="#FF0000",
        )

        left = min(vertex.x for vertex in face.bounding_poly.vertices)
        top = min(vertex.y for vertex in face.bounding_poly.vertices)
        right = max(vertex.x for vertex in face.bounding_poly.vertices)
        bottom = max(vertex.y for vertex in face.bounding_poly.vertices)
        
        cropped_face = im.crop((left, top, right, bottom))
        cropped_face.save(f"face_{i+1}.png")
    im.save(output_filename)

def crop_faces(image, faces, output_filename):
    im = Image.open(image)
    for i, face in enumerate(faces):
        box = (face['left'], face['top'], face['right'], face['bottom'])
        cropped_im = im.crop(box)
        cropped_im.save(f"{output_filename}_face_{i}.jpg")

def main(input_filename, output_filename, max_results):
    with open(input_filename, "rb") as image:
        faces = detect_face(image, max_results)
        print("Found {} face{}".format(len(faces), "" if len(faces) == 1 else "s"))

        print(f"Writing to file {output_filename}")
        # Reset the file pointer, so we can read the file again
        image.seek(0)
        highlight_faces(image, faces, output_filename)

        image.seek(0)
        crop_faces(image, faces, output_filename)

main("./input_images/HiroseSuzu_0.jpg", "output_hirose.jpg", 4)