const { spawn } = require('child_process');

// Function to execute the Python script with a single parameter
function executePythonScript(singleParam) {
    /**
     * Spawns a Python process to run the script and passes one parameter as a command-line argument.
     * 
     * Args:
     * singleParam (string): The parameter to be passed to the Python script.
     */
    if (typeof singleParam !== 'string') {
        console.error('Invalid parameter type: expected a string.');
        return;
    }

    // Spawn a new Python process using the 'child_process' module, passing one parameter
    const pythonProcess = spawn('python', ['./finetune_llama.py', singleParam]);

    // Event listener for capturing the standard output (stdout) from the Python script
    pythonProcess.stdout.on('data', (outputData) => {
        console.log(`Python Output (stdout): ${outputData}`);
    });

    // Event listener for capturing the standard error output (stderr) from the Python script
    pythonProcess.stderr.on('data', (errorData) => {
        console.error(`Python Error (stderr): ${errorData}`);
    });

    // Event listener for handling the process close event
    pythonProcess.on('close', (exitCode) => {
        if (exitCode === 0) {
            console.log('Python process completed successfully.');
        } else {
            console.error(`Python process exited with code ${exitCode}. Please check your script for errors.`);
        }
    });

    // Event listener for handling any errors that occur while spawning the Python process
    pythonProcess.on('error', (error) => {
        console.error(`Error spawning Python process: ${error.message}`);
        if (error.code === 'ENOENT') {
            console.error('Error: Python executable not found. Please ensure Python is installed and accessible in your PATH.');
        } else {
            console.error('An unexpected error occurred while starting the Python process.');
        }
    });
}

// Example usage of the function by passing a single parameter
const singleParameter = 'example_value';
// Execute the Python script with the single parameter
executePythonScript(singleParameter);
