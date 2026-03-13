import { readdir } from 'node:fs/promises';
import { join, resolve } from 'node:path';

// Secure way to get current directory regardless of module system
const currentDir = __dirname;

const folderName = process.argv[2];
const fileName = process.argv[3];

if (!folderName) {
  console.error('❌ Please provide a folder name.');
  console.error('Usage: npm start <folderName> [fileName]');
  console.error('Example (runs first .ts file): npm start S01E01');
  console.error('Example (runs specific file): npm start S01E02 findhim_ts.ts');
  process.exit(1);
}

const baseDir = resolve(currentDir, 'course', folderName);

async function run() {
  try {
    const files = await readdir(baseDir);
    let tsFile: string | undefined;

    if (fileName) {
      tsFile = files.find((f) => f === fileName || f === `${fileName}.ts`);
      if (!tsFile) {
        console.error(`❌ File '${fileName}' not found in: ${baseDir}`);
        process.exit(1);
      }
    } else {
      tsFile = files.find((f) => f.endsWith('.ts'));
      if (!tsFile) {
        console.error(`❌ No .ts file found in: ${baseDir}`);
        process.exit(1);
      }
    }

    const filePath = join(baseDir, tsFile);
    console.log(`🚀 Running ${tsFile} from ${folderName}...\n`);

    // Dynamically import the target TS file
    // file:// protocol ensures absolute paths on Windows work correctly in dynamic imports
    await import(`file://${filePath}`);
  } catch (error) {
    console.error(`❌ Could not find or read directory: ${baseDir}`);
    console.error(`Make sure the folder exists inside src/course/${folderName}`);
    process.exit(1);
  }
}

run();
