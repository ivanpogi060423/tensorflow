import React, { useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import Papa from 'papaparse';
import { Line } from 'react-chartjs-2';
import { Chart, registerables } from 'chart.js';

Chart.register(...registerables); // Register all components

const App = () => {
  const [data, setData] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      Papa.parse(file, {
        header: true,
        complete: (results) => {
          preprocessData(results.data);
        },
      });
    }
  };

  const preprocessData = (data) => {
    console.log('Raw Data:', data); // Log raw data
    
    const processedData = data.map((row) => ({
      sales_date: new Date(row.sales_date).getMonth() + 1, // Convert YYYY-MM to month number
      product_description: row.product_description === 'Product A' ? 0 : 1, // Encode products
      quantity_sold: parseFloat(row.quantity_sold),
    }));
  
    console.log('Processed Data:', processedData); // Log processed data
  
    setData(processedData);
    trainModel(processedData);
  };
  

  const trainModel = async (data) => {
    setLoading(true);
    const xs = tf.tensor2d(data.map(d => [d.sales_date, d.product_description]));
    const ys = tf.tensor2d(data.map(d => d.quantity_sold), [data.length, 1]);
  
    console.log('Training Data:', xs.arraySync(), ys.arraySync()); // Log training data
  
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 10, activation: 'relu', inputShape: [2] }));
    model.add(tf.layers.dense({ units: 1 }));
  
    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
  
    await model.fit(xs, ys, { epochs: 100 });
  
    console.log('Model trained'); // Log after training
  
    forecastSales(model);
    setLoading(false);
  };

  
  const forecastSales = (model) => {
    const futurePredictions = [];
    for (let month = 1; month <= 6; month++) {
      for (let product = 0; product <= 1; product++) {
        const input = tf.tensor2d([[month, product]]);
        const prediction = model.predict(input).dataSync()[0];
        futurePredictions.push({
          month: month,
          product: product,
          quantity_sold: prediction,
        });
      }
    }
    console.log('Future Predictions:', futurePredictions); // Log predictions
    setPredictions(futurePredictions); // Store predictions in state
  };
  
  const getChartData = () => {

    const labels = data.map(d => `Month ${d.sales_date}`); // Actual sales dates for the x-axis
    const actualData = data.map(d => d.quantity_sold);
    
    // Prepare predicted data for the next 6 months
    const predictedData = [];
    for (let month = 1; month <= 6; month++) {
      for (let product = 0; product <= 1; product++) {
        const prediction = predictions.find(p => p.month === month && p.product === product);
        predictedData.push(prediction ? prediction.quantity_sold : null); // Push predicted quantity or null if not found
        console.log('Actual Data:', actualData);
        console.log('Predicted Data:', predictedData);
      }
    }
  
    // Create labels for predicted data
    const predictedLabels = [];
    for (let month = 1; month <= 6; month++) {
      predictedLabels.push(`Month ${month} Product A`, `Month ${month} Product B`);
    }
  
    return {
      labels: [...labels, ...predictedLabels], // Combine actual and predicted labels
      datasets: [
        {
          label: 'Actual Sales',
          data: actualData,
          borderColor: 'blue',
          fill: false,
        },
        {
          label: 'Predicted Sales',
          data: predictedData,
          borderColor: 'red',
          fill: false,
        },
      ],
    };
  };


  return (
    <div>
      <h1>Sales Forecasting</h1>
      <input type="file" accept=".csv" onChange={handleFileChange} />
      {loading && <p>Loading...</p>}
      <Line 
  data={getChartData()} 
  options={{
    responsive: true,
    plugins: {
      legend: {
        display: true,
        position: 'top',
      },
      tooltip: {
        mode: 'index',
        intersect: false,
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Months',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Quantity Sold',
        },
      },
    },
  }} 
/>
    </div>
  );
};

export default App;