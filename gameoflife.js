// How many cells it contains in width and height
// Use a size that's a power of two for easier math
const GRID_SIZE = 32;

// Update every 200ms (5 times/s)
const UPDATE_INTERVAL = 200;

// There is a theoretical ideal workgroup size for every GPU, but it's dependent
// on architectural details that WebGPU doesn't expose, so usually you want to
// pick a number driven by the requirements of the shader. Lacking that, given the
// wide range of hardware that WebGPU content may run on, 64 is a good number that's
// unlikely to exceed any hardware limits but still handles large enough batches to
// be reasonably efficient. (8 x 8 == 64, so your workgroup size follows this advice.)
const WORKGROUP_SIZE = 8;

// =============================== RUN ===============================

async function run() {
    const canvas = document.querySelector("canvas");

    // Check if the user's browser can use WebGPU.
    if (!navigator.gpu) {
        throw new Error("WebGPU not supported on this browser.");
    }

    // An adapter as WebGPU's representation of a specific
    // piece of GPU hardware in your device.
    // Specify whether you want to use low-power or high-performance
    // hardware on devices with multiple GPUs 
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error("No appropriate GPUAdapter found.");
    }

    // The device is the main interface through which most
    // interaction with the GPU happens
    const device = await adapter.requestDevice();

    // Configure the canvas to be used with the device you just created
    const context = canvas.getContext("webgpu");
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: canvasFormat,
    });

    // ========== Create vertex buffer ==========

    // Square made from two triangles (GPUs work in terms of triangles)
    const vertices = new Float32Array([
        //   X,    Y,
        -0.8, -0.8, // Triangle 1
        0.8, -0.8,
        0.8,  0.8,

        -0.8, -0.8, // Triangle 2
        0.8,  0.8,
        -0.8,  0.8,
    ]);

    // GPUs frequently have their own memory that is highly optimized for rendering
    // For a lot of values, including vertex data, the GPU-side memory is
    // managed through GPUBuffer objects. A buffer is a block of memory that's easily
    // accessible to the GPU and flagged for certain purposes.
    // You want the buffer to be used for vertex data (GPUBufferUsage.VERTEX)
    // and that you also want to be able to copy data into it (GPUBufferUsage.COPY_DST).
    const vertexBuffer = device.createBuffer({
        label: "Cell vertices",
        size: vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    // When the buffer is initially created, the memory it contains will be initialized
    // to zero. There are several ways to change its contents. One way:
    device.queue.writeBuffer(vertexBuffer, 0, vertices);

    // Tell WebGPU more about the structure of the vertex data.
    const vertexBufferLayout = {
        arrayStride: 8,
        attributes: [{
            format: "float32x2",
            offset: 0,
            shaderLocation: 0, // Position, see vertex shader
        }],
    };

    // ========== Create the bind group layout ==========

    const bindGroupLayout = device.createBindGroupLayout({
        label: "Cell Bind Group Layout",
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
                buffer: {} // Grid uniform buffer
            }, 
            {
                binding: 1,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage"} // Cell state input buffer
            }, 
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage"} // Cell state output buffer
            }
        ]
    });

    // ========== Create pipeline layout ==========

    const pipelineLayout = device.createPipelineLayout({
        label: "Cell Pipeline Layout",
        bindGroupLayouts: [ bindGroupLayout ],
    });

    // ========== Create shader module ==========

    // Shaders in WebGPU are written in a shading language called WGSL (WebGPU Shading Language).
    // WGSL is, syntactically, a bit like Rust, with features aimed at making common types of GPU work
    // (like vector and matrix math) easier and faster.
    // Vertex shader: GPU calls that function once for every vertex in your vertexBuffer.
    // GPUs excel at running shaders like these in parallel, potentially processing hundreds
    // (or even thousands!) of vertices at the same time!
    // Fragment shader: they're invoked for every pixel being drawn. Fragment shaders are always called
    // after vertex shaders. The GPU takes the output of the vertex shaders and triangulates it,
    // creating triangles out of sets of three points. It then rasterizes each of those triangles by
    // figuring out which pixels of the output color attachments are included in that triangle, and then
    // calls the fragment shader once for each of those pixels. The fragment shader returns a color,
    // typically calculated from values sent to it from the vertex shader and assets like textures,
    // which the GPU writes to the color attachment.
    // vec4f(x, y, z, w) - w: w value there makes math with 4x4 matrices work, which is something that you
    // do a lot of when rendering 3D graphics, but it's rare that you need to manipulate it directly
    // You want to just leave it as 1 for this codelab.
    const cellShaderModule = device.createShaderModule({
        label: "Cell shader",
        code: `
            struct VertexInput {
                @location(0) pos: vec2f,
                @builtin(instance_index) instance: u32
            };

            struct VertexOutput {
                @builtin(position) pos: vec4f,
                @location(0) cell: vec2f,
            };

            @group(0) @binding(0) var<uniform> grid: vec2f;

            @group(0) @binding(1) var<storage> cellState: array<u32>;

            @vertex
            fn vertexMain(input: VertexInput) -> 
                VertexOutput {
                let i = f32(input.instance);
                let cell = vec2f(i % grid.x, floor(i / grid.x));
                let state = f32(cellState[input.instance]);
                
                let cellOffset = cell / grid * 2;
                let gridPos = (input.pos * state + 1) / grid - 1 + cellOffset;

                var output: VertexOutput;
                output.pos = vec4f(gridPos, 0, 1);
                output.cell = cell / grid;
                return output;
            }

            @fragment
            fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
                //return vec4f(input.cell/grid, 0, 1); // (Red, Green, Blue, Alpha)
                let c = input.cell;
                return vec4f(c, 1-c.x, 1);
            }
        `
    });

    // ========== Create rendering pipeline ==========

    // The render pipeline controls how geometry is drawn, including things like which shaders are used,
    // how to interpret data in vertex buffers, which kind of geometry should be rendered
    // (lines, points, triangles...), and more!
    const cellPipeline = device.createRenderPipeline({
        label: "Cell pipeline",
        layout: pipelineLayout,
        vertex: {
          module: cellShaderModule,
          entryPoint: "vertexMain",
          buffers: [vertexBufferLayout]
        },
        fragment: {
          module: cellShaderModule,
          entryPoint: "fragmentMain",
          targets: [{
            format: canvasFormat
          }]
        }
    });

    // ========== Create compute shader ==========

    // Compute shader that will process the game of life simulation.
    const simulationShaderModule = device.createShaderModule({
        label: "Life simulation shader",
        code: `
        @group(0) @binding(0) var<uniform> grid: vec2f;

        @group(0) @binding(1) var<storage> cellStateIn: array<u32>;
        @group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;

        fn cellIndex(cell: vec2u) -> u32 {
            return (cell.y % u32(grid.y)) * u32(grid.x) +
                (cell.x % u32(grid.x));
        }

        fn cellActive(x: u32, y: u32) -> u32 {
            return cellStateIn[cellIndex(vec2(x, y))];
        }

        @compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
        fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
            // Determine how many active neighbors this cell has.
            let activeNeighbors = cellActive(cell.x+1, cell.y+1) +
                                cellActive(cell.x+1, cell.y) +
                                cellActive(cell.x+1, cell.y-1) +
                                cellActive(cell.x, cell.y-1) +
                                cellActive(cell.x-1, cell.y-1) +
                                cellActive(cell.x-1, cell.y) +
                                cellActive(cell.x-1, cell.y+1) +
                                cellActive(cell.x, cell.y+1);

            let i = cellIndex(cell.xy);

            // Conway's game of life rules:
            switch activeNeighbors {
            case 2: { // Active cells with 2 neighbors stay active.
                cellStateOut[i] = cellStateIn[i];
            }
            case 3: { // Cells with 3 neighbors become or stay active.
                cellStateOut[i] = 1;
            }
            default: { // Cells with < 2 or > 3 neighbors become inactive.
                cellStateOut[i] = 0;
            }
            }
        }
        `
    });

    // ========== Create compute pipeline ==========

    // Compute pipeline that updates the game state.
    const simulationPipeline = device.createComputePipeline({
        label: "Simulation pipeline",
        layout: pipelineLayout,
        compute: {
        module: simulationShaderModule,
        entryPoint: "computeMain",
        }
    });

    // ========== Create uniform and storage buffers ==========

    // Create a uniform buffer that describes the grid
    const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
    const uniformBuffer = device.createBuffer({
        label: "Grid Uniforms",
        size: uniformArray.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

    // Create a storage buffer to control cell state
    // Create an array representing the active state of each cell
    const cellStateArray = new Uint32Array(GRID_SIZE * GRID_SIZE);

    // Create a storage buffer to hold the cell state.
    const cellStateStorage = [
        device.createBuffer({
            label: "Cell State A",
            size: cellStateArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        }),
        device.createBuffer({
            label: "Cell State B",
            size: cellStateArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })
    ]

    for (let i = 0; i < cellStateArray.length; ++i) {
        cellStateArray[i] = Math.random() > 0.6 ? 1 : 0;
    }
    device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);

    // ========== Create bind groups ==========

    // A bind group is a collection of resources that you want to make accessible to your shader at the
    // same time. It can include several types of buffers, like your uniform buffer, and other resources like
    // textures and samplers that are not covered here but are common parts of WebGPU rendering techniques.
    const bindGroups = [
        device.createBindGroup({
            label: "Cell renderer bind group A",
            layout: bindGroupLayout, // @group(0)
            entries: [
                {
                    binding: 0, // @binding(0)
                    resource: { buffer: uniformBuffer }
                },
                {
                    binding: 1, // @binding(1)
                    resource: { buffer: cellStateStorage[0] }
                },
                {
                    binding: 2, // @binding(2)
                    resource: { buffer: cellStateStorage[1] }
                },
            ],
        }),
        device.createBindGroup({
            label: "Cell renderer bind group B",
            layout: bindGroupLayout, // @group(0)
            entries: [
                {
                    binding: 0, // @binding(0)
                    resource: { buffer: uniformBuffer }
                },
                {
                    binding: 1, // @binding(1)
                    resource: { buffer: cellStateStorage[1] }
                },
                {
                    binding: 2, // @binding(2)
                    resource: { buffer: cellStateStorage[0] }
                },
            ],
        }),
    ]

    // ========== updateGrid ==========

    let step = 0; // Track how many simulation steps have been run

    function updateGrid() {
        // Interface for recording GPU commands
        const encoder = device.createCommandEncoder();

        // Start a compute pass
        const computePass = encoder.beginComputePass();

        computePass.setPipeline(simulationPipeline);
        computePass.setBindGroup(0, bindGroups[step % 2]);

        // The number you pass into dispatchWorkgroups() is not the number of invocations! Instead,
        // it's the number of workgroups to execute, as defined by the @workgroup_size in your shader.
        const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
        computePass.dispatchWorkgroups(workgroupCount, workgroupCount);

        computePass.end();

        step++; // Increment the step count
    
        // Render passes require that you provide a GPUTextureView instead of
        // a GPUTexture, which tells it which parts of the texture to render to
        // 'loadOp' indicates what you want to do with the texture
        // when the render pass starts
        // 'storeOp' indicates what to do with the results of any drawing
        // once the render pass is finished
        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                loadOp: "clear",
                clearValue: { r: 0, g: 0, b: 0, a: 1 }, 
                storeOp: "store",
            }]
        });

        pass.setPipeline(cellPipeline);
        pass.setVertexBuffer(0, vertexBuffer);
        pass.setBindGroup(0, bindGroups[step % 2]);
        pass.draw(vertices.length / 2, GRID_SIZE * GRID_SIZE); // 6 vertices
    
        // End the render pass
        pass.end();
    
        // The command buffer is an opaque handle to the recorded commands.
        // Create a GPUCommandBuffer, call finish() on the command encoder.
        const commandBuffer = encoder.finish();
    
        // Submit the command buffer to the GPU using the queue of the GPUDevice.
        // The queue performs all GPU commands, ensuring that their execution is
        // well ordered and properly synchronized.
        // The queue's submit() method takes in an array of command buffers,
        // though in this case you only have one.
        // Commonly combined as: device.queue.submit([encoder.finish()]);
        device.queue.submit([commandBuffer]);
    }        

    setInterval(updateGrid, UPDATE_INTERVAL);

}


run();