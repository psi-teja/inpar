/** @type {import('next').NextConfig} */
const nextConfig = {}

module.exports = {
    
  images: {
    domains: ['localhost'],
  },
    webpack: (config, { isServer }) => {
      // For binary files like canvas.node
      config.module.rules.push({
        test: /\.node$/,
        use: 'raw-loader',
      });
  
      return config;
    },
  };
  