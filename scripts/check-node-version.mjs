const SUPPORTED_MAJOR = 20;
const MIN_MINOR = 19;

function parseVersion(raw) {
  const normalized = raw.startsWith("v") ? raw.slice(1) : raw;
  const [major, minor, patch] = normalized.split(".").map((value) => Number.parseInt(value, 10));
  if ([major, minor, patch].some((value) => Number.isNaN(value))) {
    throw new Error(`Unable to parse Node version: ${raw}`);
  }
  return { major, minor, patch, normalized };
}

const current = parseVersion(process.version);
const supported =
  current.major === SUPPORTED_MAJOR &&
  (current.minor > MIN_MINOR || (current.minor === MIN_MINOR && current.patch >= 0));

if (!supported) {
  console.error(
    [
      `OceanRoute requires Node ${SUPPORTED_MAJOR}.${MIN_MINOR}.x or newer within Node ${SUPPORTED_MAJOR}.x.`,
      `Current runtime: ${current.normalized}.`,
      "Install and use Node 20.19.x before running dev, build, or test commands.",
    ].join(" "),
  );
  process.exit(1);
}
