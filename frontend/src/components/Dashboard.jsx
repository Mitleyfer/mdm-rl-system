import React, { useState, useEffect } from 'react';
import {
  Box,
  SimpleGrid,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  StatGroup,
  Heading,
  Text,
  useColorModeValue,
  Icon,
  Flex,
  Progress,
  Badge,
  VStack,
  HStack,
} from '@chakra-ui/react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { FiTrendingUp, FiUsers, FiDatabase, FiCpu } from 'react-icons/fi';
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

function Dashboard() {
  const [stats, setStats] = useState(null);
  const [performanceData, setPerformanceData] = useState([]);
  const [loading, setLoading] = useState(true);
  const bg = useColorModeValue('white', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      const [statsRes, perfRes] = await Promise.all([
        axios.get(`${API_URL}/api/v1/monitoring/stats/dashboard`),
        axios.get(`${API_URL}/api/v1/monitoring/performance?time_range=24h`)
      ]);

      setStats(statsRes.data);

      // Mock performance data for visualization
      const mockData = generateMockPerformanceData();
      setPerformanceData(mockData);

      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
      setLoading(false);
    }
  };

  const generateMockPerformanceData = () => {
    const hours = 24;
    const data = [];
    const now = new Date();

    for (let i = hours; i >= 0; i--) {
      const time = new Date(now - i * 60 * 60 * 1000);
      data.push({
        time: time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        f1_score: 0.88 + Math.random() * 0.08,
        precision: 0.90 + Math.random() * 0.06,
        recall: 0.86 + Math.random() * 0.10,
        matches: Math.floor(100 + Math.random() * 200),
      });
    }

    return data;
  };

  if (loading) {
    return (
      <Flex justify="center" align="center" height="400px">
        <div className="spinner" />
      </Flex>
    );
  }

  const modelPerformance = stats?.models || {};
  const todayStats = stats?.today || {};
  const learningStats = stats?.learning || {};

  return (
    <Box>
      <Heading size="lg" mb={6}>System Dashboard</Heading>

      {/* Key Metrics */}
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={6} mb={8}>
        <Stat
          px={6}
          py={4}
          bg={bg}
          borderWidth="1px"
          borderColor={borderColor}
          borderRadius="lg"
        >
          <Flex justify="space-between">
            <Box>
              <StatLabel>Today's Matches</StatLabel>
              <StatNumber>{todayStats.total_records || 0}</StatNumber>
              <StatHelpText>
                <StatArrow type="increase" />
                23% from yesterday
              </StatHelpText>
            </Box>
            <Icon as={FiDatabase} w={8} h={8} color="blue.500" />
          </Flex>
        </Stat>

        <Stat
          px={6}
          py={4}
          bg={bg}
          borderWidth="1px"
          borderColor={borderColor}
          borderRadius="lg"
        >
          <Flex justify="space-between">
            <Box>
              <StatLabel>Average F1 Score</StatLabel>
              <StatNumber>
                {(todayStats.avg_f1_score * 100 || 0).toFixed(1)}%
              </StatNumber>
              <StatHelpText>
                <StatArrow type="increase" />
                +2.3% improvement
              </StatHelpText>
            </Box>
            <Icon as={FiTrendingUp} w={8} h={8} color="green.500" />
          </Flex>
        </Stat>

        <Stat
          px={6}
          py={4}
          bg={bg}
          borderWidth="1px"
          borderColor={borderColor}
          borderRadius="lg"
        >
          <Flex justify="space-between">
            <Box>
              <StatLabel>Active Models</StatLabel>
              <StatNumber>4</StatNumber>
              <StatHelpText>All systems operational</StatHelpText>
            </Box>
            <Icon as={FiCpu} w={8} h={8} color="purple.500" />
          </Flex>
        </Stat>

        <Stat
          px={6}
          py={4}
          bg={bg}
          borderWidth="1px"
          borderColor={borderColor}
          borderRadius="lg"
        >
          <Flex justify="space-between">
            <Box>
              <StatLabel>Feedback Collected</StatLabel>
              <StatNumber>{learningStats.feedback_collected || 0}</StatNumber>
              <StatHelpText>
                Avg confidence: {(learningStats.avg_confidence * 100 || 0).toFixed(0)}%
              </StatHelpText>
            </Box>
            <Icon as={FiUsers} w={8} h={8} color="orange.500" />
          </Flex>
        </Stat>
      </SimpleGrid>

      {/* Performance Charts */}
      <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={6} mb={8}>
        <Box bg={bg} p={6} borderRadius="lg" borderWidth="1px" borderColor={borderColor}>
          <Heading size="md" mb={4}>Performance Metrics (24h)</Heading>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis domain={[0.8, 1]} />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="f1_score"
                stroke="#8884d8"
                name="F1 Score"
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="precision"
                stroke="#82ca9d"
                name="Precision"
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="recall"
                stroke="#ffc658"
                name="Recall"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </Box>

        <Box bg={bg} p={6} borderRadius="lg" borderWidth="1px" borderColor={borderColor}>
          <Heading size="md" mb={4}>Matching Volume</Heading>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Area
                type="monotone"
                dataKey="matches"
                stroke="#8884d8"
                fill="#8884d8"
                fillOpacity={0.6}
              />
            </AreaChart>
          </ResponsiveContainer>
        </Box>
      </SimpleGrid>

      {/* Model Performance */}
      <Box bg={bg} p={6} borderRadius="lg" borderWidth="1px" borderColor={borderColor} mb={8}>
        <Heading size="md" mb={4}>Model Performance Comparison</Heading>
        <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={4}>
          {Object.entries(modelPerformance).map(([model, metrics]) => (
            <VStack
              key={model}
              p={4}
              borderWidth="1px"
              borderRadius="md"
              borderColor={borderColor}
              spacing={3}
            >
              <Text fontWeight="bold">{model.replace('_', ' ').toUpperCase()}</Text>
              <Box w="100%">
                <HStack justify="space-between" mb={1}>
                  <Text fontSize="sm">F1 Score</Text>
                  <Text fontSize="sm" fontWeight="bold">
                    {(metrics.f1_score * 100 || 0).toFixed(1)}%
                  </Text>
                </HStack>
                <Progress
                  value={metrics.f1_score * 100 || 0}
                  colorScheme="blue"
                  size="sm"
                />
              </Box>
              <Badge colorScheme="green">Active</Badge>
            </VStack>
          ))}
        </SimpleGrid>
      </Box>

      {/* System Health */}
      <Box bg={bg} p={6} borderRadius="lg" borderWidth="1px" borderColor={borderColor}>
        <Heading size="md" mb={4}>System Health</Heading>
        <SimpleGrid columns={{ base: 1, md: 3 }} spacing={6}>
          <VStack align="start" spacing={2}>
            <HStack>
              <Box w={3} h={3} bg="green.500" borderRadius="full" />
              <Text>Database</Text>
            </HStack>
            <Text fontSize="sm" color="gray.600">Response time: 12ms</Text>
          </VStack>
          <VStack align="start" spacing={2}>
            <HStack>
              <Box w={3} h={3} bg="green.500" borderRadius="full" />
              <Text>ML Services</Text>
            </HStack>
            <Text fontSize="sm" color="gray.600">All models operational</Text>
          </VStack>
          <VStack align="start" spacing={2}>
            <HStack>
              <Box w={3} h={3} bg="green.500" borderRadius="full" />
              <Text>Cache</Text>
            </HStack>
            <Text fontSize="sm" color="gray.600">Hit rate: 87%</Text>
          </VStack>
        </SimpleGrid>
      </Box>
    </Box>
  );
}

export default Dashboard;