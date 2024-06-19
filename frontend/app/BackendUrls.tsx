// BackendUrls.tsx

const localUrls = {
  getSubTableUrl: `http://localhost:8000/db_connect/get/SubTable`,
  uploadDocUrl: `http://localhost:8000/db_connect/upload/doc/`,
  uploadGTJsonUrl: `http://localhost:8000/db_connect/upload/json/gt_json`,
  getJsonUrl: `http://localhost:8000/db_connect/get/json`,
  getDocUrl: `http://localhost:8000/db_connect/get/document`,
};

const awsUrls = {
  getSubTableUrl: `https://o4xsvxdeo7p74yfscmfplihqqm0spsqn.lambda-url.ap-south-1.on.aws/get/SubTable`,
  uploadDocUrl: `https://loqc5abj3t6z2crtwezyyxcsae0zsxmz.lambda-url.ap-south-1.on.aws/upload/doc`,
  uploadGTJsonUrl: `https://loqc5abj3t6z2crtwezyyxcsae0zsxmz.lambda-url.ap-south-1.on.aws/upload/gt_json`,
  getJsonUrl: `https://abtzn7vmc5bae2auqjgtekhpiq0mehbs.lambda-url.ap-south-1.on.aws/get`,
  getDocUrl: `https://abtzn7vmc5bae2auqjgtekhpiq0mehbs.lambda-url.ap-south-1.on.aws/get/document`,
};

const BACKEND_URLS = process.env.REACT_APP_ENV === 'aws' ? awsUrls : localUrls;

export default BACKEND_URLS;
