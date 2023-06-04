import React, { useRef, useEffect } from 'react';
import Chart from 'chart.js/auto';

const LineChart = () => {
  const chartRef = useRef(null);
  let chart = null;

  useEffect(() => {
    if (!chart) {
      chart = new Chart(chartRef.current, {
        type: 'line',
        data: {
          labels: ['January', 'February', 'March', 'April', 'May', 'June', 'July'],
          datasets: [{
            label: 'My Dataset',
            data: [10, 20, 30, 40, 50, 60, 70],
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 1
          }]
        },
        options: {
          scales: {
            yAxes: [{
              ticks: {
                beginAtZero: true
              }
            }]
          }
        }
      });
    }

    return () => {
      chart.destroy();
      chart = null;
    };
  }, []);

  return (
    <canvas id="myChart" ref={chartRef} />
  );
};

export default LineChart;
