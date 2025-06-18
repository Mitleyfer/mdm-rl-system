import React, { useState, useEffect } from 'react';
import {
  ChakraProvider,
  Box,
  VStack,
  HStack,
  Heading,
  Text,
  Button,
  Input,
  Select,
  Progress,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Alert,
  AlertIcon,
  Tab,
  Tabs,
  TabList,
  TabPanel,
  TabPanels,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  SimpleGrid,
  Container,
  Divider,
  Badge,
  useToast,
  IconButton,
  Drawer,
  DrawerBody,
  DrawerHeader,
  DrawerOverlay,
  DrawerContent,
  DrawerCloseButton,
  useDisclosure,
  FormControl,
  FormLabel,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  Switch,
  Code,
  Link
} from '@chakra-ui/react';
import {
  FiUpload,
  FiSettings,
  FiActivity,
  FiDatabase,
  FiCheck,
  FiX,
  FiRefreshCw
} from 'react-icons/fi';
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [rules, setRules] = useState({
    name_threshold: 0.85,
    address_threshold: 0.80,
    phone_threshold: 0.95,
    email_threshold: 0.98,
    fuzzy_weight: 0.7,
    exact_weight: 0.3,
    enable_phonetic: true,
    enable_abbreviation: true,
    blocking_key: 'sorted_neighborhood'
  });
  const [models, setModels] = useState({});
  const toast = useToast();
  const { isOpen, onOpen, onClose } = useDisclosure();

  useEffect(() => {
    fetchDatasets();
    fetchModels();
    const interval = setInterval(() => {
      if (selectedDataset && selectedDataset.status === 'processing') {
        checkDatasetStatus(selectedDataset.id);
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [selectedDataset]);

  const fetchDatasets = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/v1/datasets`);
      setDatasets(response.data);
    } catch (error) {
      console.error('Failed to fetch datasets:', error);
    }
  };

  const fetchModels = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/v1/models/status`);
      setModels(response.data);
    } catch (error) {
      console.error('Failed to fetch models:', error);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('dataset_name', file.name);
    formData.append('dataset_type', 'customer'); // Default

    setIsUploading(true);
    setUploadProgress(0);

    try {
      const response = await axios.post(`${API_URL}/api/v1/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(progress);
        },
      });

      toast({
        title: 'Dataset uploaded successfully',
        description: `Processing dataset: ${response.data.dataset_id}`,
        status: 'success',
        duration: 5000,
        isClosable: true,
      });

      setSelectedDataset(response.data);
      fetchDatasets();
    } catch (error) {
      toast({
        title: 'Upload failed',
        description: error.response?.data?.detail || 'An error occurred',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const checkDatasetStatus = async (datasetId) => {
    try {
      const response = await axios.get(`${API_URL}/api/v1/status/${datasetId}`);
      setSelectedDataset(response.data);
      if (response.data.status === 'completed' && response.data.results) {
        setMetrics(response.data.results.final_performance);
      }
    } catch (error) {
      console.error('Failed to check status:', error);
    }
  };

  const updateRules = async () => {
    try {
      const response = await axios.post(`${API_URL}/api/v1/models/update_rules`, rules);
      toast({
        title: 'Rules updated successfully',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
      onClose();
    } catch (error) {
      toast({
        title: 'Failed to update rules',
        description: error.response?.data?.detail || 'An error occurred',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  return (
    <ChakraProvider>
      <Container maxW="container.xl" py={8}>
        <VStack spacing={8} align="stretch">
          {/* Header */}
          <Box>
            <Heading size="xl" mb={2}>
              MDM Reinforcement Learning System
            </Heading>
            <Text color="gray.600">
              Adaptive Data Matching Rules Management using Multi-Paradigm RL
            </Text>
          </Box>

          {/* Model Status */}
          <Box>
            <Heading size="md" mb={4}>
              <HStack>
                <FiActivity />
                <Text>Model Status</Text>
              </HStack>
            </Heading>
            <SimpleGrid columns={{ base: 1, md: 4 }} spacing={4}>
              {Object.entries(models.active_agents || {}).map(([agent, status]) => (
                <Stat key={agent} border="1px" borderColor="gray.200" p={4} borderRadius="md">
                  <StatLabel>{agent.replace('_', ' ').toUpperCase()}</StatLabel>
                  <StatNumber fontSize="md">
                    {status === 'healthy' ? (
                      <Badge colorScheme="green">Active</Badge>
                    ) : (
                      <Badge colorScheme="red">Error</Badge>
                    )}
                  </StatNumber>
                </Stat>
              ))}
            </SimpleGrid>
          </Box>

          {/* Upload Section */}
          <Box>
            <Heading size="md" mb={4}>
              <HStack>
                <FiDatabase />
                <Text>Dataset Upload</Text>
              </HStack>
            </Heading>
            <VStack align="stretch" spacing={4}>
              <HStack>
                <Input
                  type="file"
                  accept=".csv,.json,.xlsx,.xls"
                  onChange={handleFileUpload}
                  disabled={isUploading}
                  display="none"
                  id="file-upload"
                />
                <Button
                  as="label"
                  htmlFor="file-upload"
                  leftIcon={<FiUpload />}
                  colorScheme="blue"
                  isLoading={isUploading}
                  loadingText="Uploading..."
                >
                  Upload Dataset
                </Button>
                <Button
                  leftIcon={<FiSettings />}
                  onClick={onOpen}
                  variant="outline"
                >
                  Configure Rules
                </Button>
              </HStack>

              {uploadProgress > 0 && uploadProgress < 100 && (
                <Progress value={uploadProgress} size="sm" colorScheme="blue" />
              )}

              {selectedDataset && (
                <Alert status={
                  selectedDataset.status === 'completed' ? 'success' :
                  selectedDataset.status === 'failed' ? 'error' : 'info'
                }>
                  <AlertIcon />
                  Dataset: {selectedDataset.dataset_id} - Status: {selectedDataset.status}
                  {selectedDataset.progress !== undefined && ` (${selectedDataset.progress}%)`}
                </Alert>
              )}
            </VStack>
          </Box>

          {/* Metrics Display */}
          {metrics && (
            <Box>
              <Heading size="md" mb={4}>Performance Metrics</Heading>
              <SimpleGrid columns={{ base: 1, md: 3 }} spacing={4}>
                <Stat border="1px" borderColor="gray.200" p={4} borderRadius="md">
                  <StatLabel>Precision</StatLabel>
                  <StatNumber>{(metrics.precision * 100).toFixed(2)}%</StatNumber>
                  <StatHelpText>
                    <StatArrow type="increase" />
                    {((metrics.precision - 0.71) * 100).toFixed(1)}%
                  </StatHelpText>
                </Stat>
                <Stat border="1px" borderColor="gray.200" p={4} borderRadius="md">
                  <StatLabel>Recall</StatLabel>
                  <StatNumber>{(metrics.recall * 100).toFixed(2)}%</StatNumber>
                  <StatHelpText>
                    <StatArrow type="increase" />
                    {((metrics.recall - 0.65) * 100).toFixed(1)}%
                  </StatHelpText>
                </Stat>
                <Stat border="1px" borderColor="gray.200" p={4} borderRadius="md">
                  <StatLabel>F1 Score</StatLabel>
                  <StatNumber>{(metrics.f1_score * 100).toFixed(2)}%</StatNumber>
                  <StatHelpText>
                    <StatArrow type="increase" />
                    {((metrics.f1_score - 0.68) * 100).toFixed(1)}%
                  </StatHelpText>
                </Stat>
              </SimpleGrid>
            </Box>
          )}

          {/* Recent Datasets */}
          <Box>
            <Heading size="md" mb={4}>Recent Datasets</Heading>
            {datasets.length > 0 ? (
              <Table variant="simple">
                <Thead>
                  <Tr>
                    <Th>Dataset Name</Th>
                    <Th>Type</Th>
                    <Th>Records</Th>
                    <Th>Status</Th>
                    <Th>F1 Score</Th>
                    <Th>Actions</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  {datasets.map((dataset) => (
                    <Tr key={dataset.id}>
                      <Td>{dataset.name}</Td>
                      <Td>{dataset.type}</Td>
                      <Td>{dataset.records_count}</Td>
                      <Td>
                        <Badge colorScheme={
                          dataset.status === 'completed' ? 'green' :
                          dataset.status === 'failed' ? 'red' : 'yellow'
                        }>
                          {dataset.status}
                        </Badge>
                      </Td>
                      <Td>
                        {dataset.results?.final_performance?.f1_score
                          ? `${(dataset.results.final_performance.f1_score * 100).toFixed(2)}%`
                          : '-'}
                      </Td>
                      <Td>
                        <IconButton
                          icon={<FiRefreshCw />}
                          size="sm"
                          onClick={() => checkDatasetStatus(dataset.id)}
                          aria-label="Refresh"
                        />
                      </Td>
                    </Tr>
                  ))}
                </Tbody>
              </Table>
            ) : (
              <Text color="gray.500">No datasets uploaded yet</Text>
            )}
          </Box>
        </VStack>

        {/* Rules Configuration Drawer */}
        <Drawer isOpen={isOpen} placement="right" onClose={onClose} size="md">
          <DrawerOverlay />
          <DrawerContent>
            <DrawerCloseButton />
            <DrawerHeader>Configure Matching Rules</DrawerHeader>
            <DrawerBody>
              <VStack spacing={6} align="stretch">
                <FormControl>
                  <FormLabel>Name Threshold: {rules.name_threshold}</FormLabel>
                  <Slider
                    value={rules.name_threshold}
                    onChange={(value) => setRules({...rules, name_threshold: value})}
                    min={0.5}
                    max={1}
                    step={0.05}
                  >
                    <SliderTrack>
                      <SliderFilledTrack />
                    </SliderTrack>
                    <SliderThumb />
                  </Slider>
                </FormControl>

                <FormControl>
                  <FormLabel>Address Threshold: {rules.address_threshold}</FormLabel>
                  <Slider
                    value={rules.address_threshold}
                    onChange={(value) => setRules({...rules, address_threshold: value})}
                    min={0.5}
                    max={1}
                    step={0.05}
                  >
                    <SliderTrack>
                      <SliderFilledTrack />
                    </SliderTrack>
                    <SliderThumb />
                  </Slider>
                </FormControl>

                <FormControl>
                  <FormLabel>Phone Threshold: {rules.phone_threshold}</FormLabel>
                  <Slider
                    value={rules.phone_threshold}
                    onChange={(value) => setRules({...rules, phone_threshold: value})}
                    min={0.7}
                    max={1}
                    step={0.05}
                  >
                    <SliderTrack>
                      <SliderFilledTrack />
                    </SliderTrack>
                    <SliderThumb />
                  </Slider>
                </FormControl>

                <FormControl>
                  <FormLabel>Email Threshold: {rules.email_threshold}</FormLabel>
                  <Slider
                    value={rules.email_threshold}
                    onChange={(value) => setRules({...rules, email_threshold: value})}
                    min={0.8}
                    max={1}
                    step={0.02}
                  >
                    <SliderTrack>
                      <SliderFilledTrack />
                    </SliderTrack>
                    <SliderThumb />
                  </Slider>
                </FormControl>

                <Divider />

                <FormControl>
                  <FormLabel>Fuzzy Weight: {rules.fuzzy_weight}</FormLabel>
                  <Slider
                    value={rules.fuzzy_weight}
                    onChange={(value) => setRules({...rules, fuzzy_weight: value})}
                    min={0}
                    max={1}
                    step={0.1}
                  >
                    <SliderTrack>
                      <SliderFilledTrack />
                    </SliderTrack>
                    <SliderThumb />
                  </Slider>
                </FormControl>

                <FormControl>
                  <FormLabel>Exact Weight: {rules.exact_weight}</FormLabel>
                  <Slider
                    value={rules.exact_weight}
                    onChange={(value) => setRules({...rules, exact_weight: value})}
                    min={0}
                    max={1}
                    step={0.1}
                  >
                    <SliderTrack>
                      <SliderFilledTrack />
                    </SliderTrack>
                    <SliderThumb />
                  </Slider>
                </FormControl>

                <Divider />

                <FormControl display="flex" alignItems="center">
                  <FormLabel mb="0">Enable Phonetic Matching</FormLabel>
                  <Switch
                    isChecked={rules.enable_phonetic}
                    onChange={(e) => setRules({...rules, enable_phonetic: e.target.checked})}
                  />
                </FormControl>

                <FormControl display="flex" alignItems="center">
                  <FormLabel mb="0">Enable Abbreviation Expansion</FormLabel>
                  <Switch
                    isChecked={rules.enable_abbreviation}
                    onChange={(e) => setRules({...rules, enable_abbreviation: e.target.checked})}
                  />
                </FormControl>

                <FormControl>
                  <FormLabel>Blocking Strategy</FormLabel>
                  <Select
                    value={rules.blocking_key}
                    onChange={(e) => setRules({...rules, blocking_key: e.target.value})}
                  >
                    <option value="sorted_neighborhood">Sorted Neighborhood</option>
                    <option value="soundex">Soundex</option>
                    <option value="first_letter">First Letter</option>
                    <option value="exact">Exact Field</option>
                  </Select>
                </FormControl>

                <Button colorScheme="blue" onClick={updateRules} size="lg">
                  Apply Rules
                </Button>
              </VStack>
            </DrawerBody>
          </DrawerContent>
        </Drawer>
      </Container>
    </ChakraProvider>
  );
}

export default App;