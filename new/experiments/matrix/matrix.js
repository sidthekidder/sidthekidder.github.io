const canvas = document.getElementById('gl-canvas');
const gl = canvas.getContext('webgl2');

if (!gl) {
  document.body.innerHTML =
    '<div class="fallback-message">This experiment needs WebGL2, which your browser doesn\'t support. Try a recent Chrome, Firefox, or Safari.</div>';
} else {
  const vertexSource = `#version 300 es
  layout(location = 0) in vec2 a_position;
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
  }`;

  const fragmentSource = `#version 300 es
  precision highp float;
  uniform vec2 u_resolution;
  uniform float u_time;
  uniform vec2 u_mouse;
  out vec4 outColor;

  float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
  }

  void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    float cols = 60.0;
    float col = floor(uv.x * cols);
    float speed = 0.3 + hash(vec2(col, 0.0)) * 0.7;
    float offset = hash(vec2(col, 1.0)) * 10.0;
    float y = fract(uv.y + u_time * speed + offset);

    float dist = length(uv - u_mouse);
    float mouseGlow = smoothstep(0.25, 0.0, dist);

    float charNoise = step(0.5, hash(vec2(col, floor((uv.y + u_time * speed) * 40.0))));
    float brightness = pow(1.0 - y, 4.0) * charNoise;
    brightness += mouseGlow * 0.3;

    vec3 green = vec3(0.1, 1.0, 0.4) * brightness;
    outColor = vec4(green, 1.0);
  }`;

  function compileShader(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error(gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  }

  const vertexShader = compileShader(gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = compileShader(gl.FRAGMENT_SHADER, fragmentSource);

  if (!vertexShader || !fragmentShader) {
    document.body.innerHTML =
      '<div class="fallback-message">This experiment needs WebGL2, which your browser doesn\'t support. Try a recent Chrome, Firefox, or Safari.</div>';
  } else {
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error(gl.getProgramInfoLog(program));
      document.body.innerHTML =
        '<div class="fallback-message">This experiment needs WebGL2, which your browser doesn\'t support. Try a recent Chrome, Firefox, or Safari.</div>';
    } else {
      const positions = new Float32Array([-1, -1, 3, -1, -1, 3]);
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      const buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

      const resolutionLoc = gl.getUniformLocation(program, 'u_resolution');
      const timeLoc = gl.getUniformLocation(program, 'u_time');
      const mouseLoc = gl.getUniformLocation(program, 'u_mouse');

      let mouseX = 0.5;
      let mouseY = 0.5;

      window.addEventListener('mousemove', (e) => {
        mouseX = e.clientX / window.innerWidth;
        mouseY = 1.0 - e.clientY / window.innerHeight;
      });

      function resize() {
        canvas.width = window.innerWidth * window.devicePixelRatio;
        canvas.height = window.innerHeight * window.devicePixelRatio;
        gl.viewport(0, 0, canvas.width, canvas.height);
      }
      window.addEventListener('resize', resize);
      resize();

      function render(timeMs) {
        const time = timeMs * 0.001;
        gl.useProgram(program);
        gl.bindVertexArray(vao);
        gl.uniform2f(resolutionLoc, canvas.width, canvas.height);
        gl.uniform1f(timeLoc, time);
        gl.uniform2f(mouseLoc, mouseX, mouseY);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        requestAnimationFrame(render);
      }
      requestAnimationFrame(render);
    }
  }
}
