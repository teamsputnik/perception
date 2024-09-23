import cv2
import numpy as np
import OpenGL.GL as gl
import xr  # Assuming you have a Python binding for OpenXR

# Load the image
image = cv2.imread('C:/Users/15105/Desktop/example2.png')
print(image.shape)
height, width, _ = image.shape

# Calculate the halfway point to split the image vertically
half_width = width // 2

# Split the image into two parts (left and right)
left_image = image[:, :half_width]   # Left part of the image
right_image = image[:, half_width:]  # Right part of the image

# Flip the images vertically because OpenGL's texture origin is bottom-left
left_image = cv2.flip(left_image, 0)/255
right_image = cv2.flip(right_image, 0)/255
print(np.max(left_image), np.min(left_image))


# Now, within your OpenXR context and rendering loop:
with xr.ContextObject(
    instance_create_info=xr.InstanceCreateInfo(
        enabled_extension_names=[
            # A graphics extension is mandatory (without a headless extension)
            xr.KHR_OPENGL_ENABLE_EXTENSION_NAME,
        ],
    ),
) as context:
    # **OpenGL context is now active**
    print("session_created")
    # Create OpenGL textures for the left and right images
    def create_texture(image_data):
        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGB,
            image_data.shape[1],
            image_data.shape[0],
            0,
            gl.GL_BGR,
            gl.GL_FLOAT,
            image_data
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        return texture_id

    print("texture_generated")
    # Set up a simple shader program (vertex and fragment shaders)
    vertex_shader_source = """
    #version 330 core
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec2 aTexCoord;

    out vec2 TexCoord;

    void main(){
        gl_Position = vec4(aPos, 1.0);
        TexCoord = aTexCoord;
    }
    """

    fragment_shader_source = """
    #version 330 core
    out vec4 FragColor;

    in vec2 TexCoord;

    uniform sampler2D ourTexture;

    void main(){
        FragColor = texture(ourTexture, TexCoord);
    }
    """

    # Compile shaders and link them into a program
    def compile_shader(source, shader_type):
        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)
        # Check for compilation errors
        if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
            infoLog = gl.glGetShaderInfoLog(shader)
            raise RuntimeError(f"Shader compilation failed:\n{infoLog.decode()}")
        return shader

    vertex_shader = compile_shader(vertex_shader_source, gl.GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_shader_source, gl.GL_FRAGMENT_SHADER)

    shader_program = gl.glCreateProgram()
    gl.glAttachShader(shader_program, vertex_shader)
    gl.glAttachShader(shader_program, fragment_shader)
    gl.glLinkProgram(shader_program)
    # Check for linking errors
    if not gl.glGetProgramiv(shader_program, gl.GL_LINK_STATUS):
        infoLog = gl.glGetProgramInfoLog(shader_program)
        raise RuntimeError(f"Program linking failed:\n{infoLog.decode()}")

    gl.glDeleteShader(vertex_shader)
    gl.glDeleteShader(fragment_shader)

    # Set up vertex data and buffers and configure vertex attributes
    vertices = np.array([
        # positions       # texture coords
        -1.0,  1.0, 0.5,   0.0, 1.0,  # Top-left
        -1.0, -1.0, 0.5,   0.0, 0.0,  # Bottom-left
         1.0, -1.0, 0.5,   1.0, 0.0,  # Bottom-right
         1.0,  1.0, 0.5,   1.0, 1.0   # Top-right
    ], dtype=np.float32)

    indices = np.array([
        0, 1, 2,  # First triangle
        0, 2, 3   # Second triangle
    ], dtype=np.uint32)

    VAO = gl.glGenVertexArrays(1)
    VBO = gl.glGenBuffers(1)
    EBO = gl.glGenBuffers(1)

    gl.glBindVertexArray(VAO)

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, VBO)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, EBO)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

    # Position attribute
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 5 * vertices.itemsize, None)
    gl.glEnableVertexAttribArray(0)
    # Texture coord attribute
    gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 5 * vertices.itemsize, gl.ctypes.c_void_p(3 * vertices.itemsize))
    gl.glEnableVertexAttribArray(1)

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
    gl.glBindVertexArray(0)
    print("start")
    for frame_index, frame_state in enumerate(context.frame_loop()):
        if frame_index%100 == 0:
            print(frame_index) 
        for view_index, view in enumerate(context.view_loop(frame_state)):
            if view_index == 1:
                # continue
                pass
            # Bind the framebuffer for the current view
            #context.bind_framebuffer(view)

            # Clear the color buffer
            gl.glClearColor(0.0, 0.0, 0.0, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            # Use the shader program
            gl.glUseProgram(shader_program)

            # Bind the appropriate texture
            if view_index == 0:
                # Left eye
                gl.glActiveTexture(gl.GL_TEXTURE0)
                left_texture = create_texture(left_image)
                gl.glBindTexture(gl.GL_TEXTURE_2D, left_texture)
            else:
                # Right eye
                gl.glActiveTexture(gl.GL_TEXTURE0)      
                right_texture = create_texture(right_image)
                gl.glBindTexture(gl.GL_TEXTURE_2D, right_texture)

            # Set the sampler uniform
            gl.glUniform1i(gl.glGetUniformLocation(shader_program, "ourTexture"), 0)

            # Render the quad
            gl.glBindVertexArray(VAO)
            gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)
            gl.glBindVertexArray(0)

            # Unbind the texture
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

            # Swap buffers or submit the frame if necessary
            #context.release_framebuffer()

    # Clean up (optional, if your application does not exit immediately)
    gl.glDeleteVertexArrays(1, [VAO])
    gl.glDeleteBuffers(1, [VBO])
    gl.glDeleteBuffers(1, [EBO])
    gl.glDeleteProgram(shader_program)
    gl.glDeleteTextures([left_texture])
    gl.glDeleteTextures([right_texture])
