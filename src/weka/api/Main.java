package weka.api;

import java.io.File;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.supervised.instance.StratifiedRemoveFolds;

public class Main {

	public static void main(String[] args) throws Exception {

		String currentDir = System.getProperty("user.dir");
		String csvSource = currentDir + "/data/stroke-dataset.csv";
		Instances dts = getInstancesFromCSV(csvSource);
//		checkData(dts, 5);
		HashMap<String, HashSet<String>> colsList = grabColNames(dts, 10, 20);
		HashSet<String> numCols = colsList.get("num_cols");
		if (numCols != null) {
			numCols.forEach(col -> {
				try {
					checkOutlier(dts, col);
					replaceWithThreshold(dts, col);
				} catch (Exception e) {
					e.printStackTrace();
				}
			});
		}
		missingValuesTable(dts, false);
		meanFill(dts, "bmi");
		missingValuesTable(dts, false);

		double[] bmiBins = { 0, 19, 25, 30, 10000 };
		String[] bmiLabels = { "Underweight", "Ideal", "Overweight", "Obesity" };

		double[] ageBins = { 0, 13, 18, 45, 60, 200 };
		String[] ageLabels = { "Children", "Teens", "Adults", "Mid Adults", "Elderly" };

		double[] glucoseBins = { 0, 90, 160, 230, 500 };
		String[] glucoseLabels = { "Low", "Normal", "High", "Very High" };

		// Create new attributes for each feature
		Attribute bmiCat = new Attribute("bmi_cat", Arrays.asList(bmiLabels));
		Attribute ageCat = new Attribute("age_cat", Arrays.asList(ageLabels));
		Attribute glucoseCat = new Attribute("glucose_cat", Arrays.asList(glucoseLabels));

		// Add the new attributes to the existing Instances object
		dts.insertAttributeAt(bmiCat, dts.numAttributes());
		dts.insertAttributeAt(ageCat, dts.numAttributes());
		dts.insertAttributeAt(glucoseCat, dts.numAttributes());

		// Iterate through each instance and add the new feature values
		for (int i = 0; i < dts.numInstances(); i++) {
			DenseInstance inst = (DenseInstance) dts.instance(i);

			double bmiValue = inst.value(dts.attribute("bmi"));
			String bmiCatValue = getLabel(bmiValue, bmiBins, bmiLabels);
			inst.setValue(dts.attribute("bmi_cat"), bmiCatValue);

			double ageValue = inst.value(dts.attribute("age"));
			String ageCatValue = getLabel(ageValue, ageBins, ageLabels);
			inst.setValue(dts.attribute("age_cat"), ageCatValue);

			double glucoseValue = inst.value(dts.attribute("avg_glucose_level"));
			String glucoseCatValue = getLabel(glucoseValue, glucoseBins, glucoseLabels);
			inst.setValue(dts.attribute("glucose_cat"), glucoseCatValue);
		}

		HashMap<String, HashSet<String>> colsListAfterAdded = grabColNames(dts, 10, 20);
		HashSet<String> binaryCols = colsListAfterAdded.get("binary_cols");
		HashSet<String> oheCols = colsListAfterAdded.get("ohe_cols");
		if (binaryCols != null) {
			binaryCols.forEach(col -> {
				try {
					le(dts, col);
				} catch (Exception e) {
					e.printStackTrace();
				}
			});
		}
		if (oheCols != null) {
			oheCols.forEach(col -> {
				try {
					ohe(dts, col);
				} catch (Exception e) {
					e.printStackTrace();
				}
			});
		}
		if (numCols != null) {
			numCols.forEach(col -> {
				try {
					standardScaler(dts, col);
				} catch (Exception e) {
					e.printStackTrace();
				}
			});
		}
		
		setClassIndex(dts, "stroke");

		Instances filteredDts = getInstExceptAttr(dts, new String[] { "id" });

		Instances[] trainTestSplit = trainTestSplit(filteredDts, 5);
		Instances trainDts = trainTestSplit[0];
		Instances testDts = trainTestSplit[1];

		Instances resampledTrainDts = oversampling(trainDts);

		kFoldsCrossValidationEvaluateModelDetails(new RandomForest(), resampledTrainDts, 10);
		kFoldsCrossValidationEvaluateModelDetails(new J48(), resampledTrainDts, 10);

	}

	static void writeARFF(Instances dts, String des) throws Exception {
		ArffSaver arffSaver = new ArffSaver();
		arffSaver.setInstances(dts);
		arffSaver.setFile(new File(des));
		arffSaver.writeBatch();
	}

	static Instances getInstancesFromARFF(String src) throws Exception {
		DataSource ds = new DataSource(src);
		return ds.getDataSet();
	}

	static Instances getInstancesFromCSV(String src) throws Exception {
		File csvFile = new File(src);
		CSVLoader csvLoader = new CSVLoader();
		csvLoader.setSource(csvFile);
		return csvLoader.getDataSet();
	}

	static void setClassIndex(Instances dts, String colName) {
		int indexToSet = dts.attribute(colName).index();
		if (dts.classIndex() != indexToSet) {
			dts.setClassIndex(indexToSet);
		}
	}

	static void fillingMissing(Instances dts, String val) {
		for (int i = 0; i < dts.numInstances(); i++) {
			Instance inst = dts.instance(i);
			for (int j = 0; j < inst.numAttributes(); j++) {
				Attribute attr = dts.attribute(j);
				if (attr.isString() && inst.stringValue(j).equals(val)) {
					inst.setMissing(j);
				}
			}
		}
	}

	static void printColNames(String title, HashSet<String> cols) {
		System.out.print(title + ": [ ");
		for (String value : cols) {
			System.out.print(value + " ");
		}
		System.out.println("] length = " + cols.size());
	}

	static HashMap<String, HashSet<String>> grabColNames(Instances dts, int catTh, int carTh) {
		HashSet<String> catCols = new HashSet<String>();
		HashSet<String> numButCat = new HashSet<String>();
		HashSet<String> catButCar = new HashSet<String>();
		HashSet<String> numCols = new HashSet<String>();
		HashSet<String> binaryCols = new HashSet<String>();
		HashSet<String> oheCols = new HashSet<String>();
		for (int i = 0; i < dts.numAttributes(); i++) {
			Attribute attr = dts.attribute(i);
			String colName = attr.name();
			AttributeStats stats = dts.attributeStats(attr.index());
			int attrNumDistinct = stats.distinctCount;

			if (attrNumDistinct == 2) {
				binaryCols.add(colName);
			}
			if (attrNumDistinct > 2 && attrNumDistinct <= 10) {
				oheCols.add(colName);
			}
			if (attr.isNominal()) {
				catCols.add(colName);
			}
			if (attr.isNumeric()) {
				numCols.add(colName);
			}
			if (attr.isNumeric() && attrNumDistinct < catTh) {
				numButCat.add(colName);
			}
			if (attr.isNominal() && attrNumDistinct > carTh) {
				catButCar.add(colName);
			}
		}
		catCols.addAll(numButCat);
		Iterator<String> catColsIteration = catCols.iterator();
		while (catColsIteration.hasNext()) {
			String col = catColsIteration.next();
			if (catButCar.contains(col)) {
				catColsIteration.remove();
			}
		}
		Iterator<String> numColsIteration = numCols.iterator();
		while (numColsIteration.hasNext()) {
			String col = numColsIteration.next();
			if (numButCat.contains(col)) {
				numColsIteration.remove();
			}
		}
		System.out.println("--------------------Grab cols name--------------------");
		printColNames("Cat_cols", catCols);
		printColNames("Num_cols", numCols);
		printColNames("Cat_but_car", catButCar);
		printColNames("Num_but_cat", numButCat);
		printColNames("Binary_cols", binaryCols);
		printColNames("Ohe_cols", oheCols);
		HashMap<String, HashSet<String>> map = new HashMap<String, HashSet<String>>();
		map.put("cat_cols", catCols);
		map.put("num_cols", numCols);
		map.put("cat_but_car", catButCar);
		map.put("binary_cols", binaryCols);
		map.put("ohe_cols", oheCols);
		return map;
	}

	static void checkData(Instances dataset, int head) throws Exception {
		System.out.println(
				"===========================================Check data===========================================");
		System.out.println("--------------------Information--------------------");
		System.out.println(dataset.toSummaryString());
		System.out.println("--------------------The First " + head + " Data--------------------");
		for (int i = 0; i < head; i++)
			System.out.println(dataset.instance(i));
		System.out.println("--------------------Missing Values--------------------");
		for (int i = 0; i < dataset.numAttributes(); i++)
			System.out.println(dataset.attribute(i).name() + ": " + dataset.attributeStats(i).missingCount);
		System.out.println("--------------------Describe the Data--------------------");
	}

	static double[] outlierTh(Instances dts, String colName, double q1, double q3) throws Exception {
		Attribute attr = dts.attribute(colName);
		Instances copiedDts = new Instances(dts);
		copiedDts.sort(attr);
		int q1Idx = (int) Math.round(q1 * copiedDts.numInstances()) - 1;
		int q3Idx = (int) Math.round(q3 * copiedDts.numInstances()) - 1;
		double q1Val = copiedDts.instance(q1Idx).value(attr.index());
		double q3Val = copiedDts.instance(q3Idx).value(attr.index());
		double interquantile_range = q3Val - q1Val;
		double up_limit = q3Val + 1.5 * interquantile_range;
		double low_limit = q1Val - 1.5 * interquantile_range;
		double[] limits = { low_limit, up_limit };
		return limits;
	}

	static boolean checkOutlier(Instances dts, String colName) throws Exception {
		double[] outlier = outlierTh(dts, colName, 0.05, 0.95);
		double lowLim = outlier[0];
		double upLim = outlier[1];
		int attrIndex = dts.attribute(colName).index();
		System.out.println(
				"===========================================Check outlier===========================================");
		for (int i = 0; i < dts.numInstances(); i++) {
			double value = dts.instance(i).value(attrIndex);
			if (value > upLim || value < lowLim) {
				System.out.println(colName + " has an outlier value");
				return true;
			}
		}
		System.out.println(colName + " does have an outlier value");
		return false;
	}

	static void replaceWithThreshold(Instances dts, String colName) throws Exception {
		double[] outlier = outlierTh(dts, colName, 0.05, 0.95);
		double lowLim = outlier[0];
		double upLim = outlier[1];
		int attrIndex = dts.attribute(colName).index();
		for (int i = 0; i < dts.numInstances(); i++) {
			Instance inst = dts.instance(i);
			double value = inst.value(attrIndex);
			if (value > upLim) {
				inst.setValue(attrIndex, upLim);
			} else if (value < lowLim) {
				inst.setValue(attrIndex, lowLim);
			}
		}
	}

	static void missingValuesTable(Instances dts, boolean na_name) {
		System.out.println("--------------------Missing values details--------------------");
		int numInst = dts.numInstances();
		boolean hasMissing = false;
		for (int i = 0; i < dts.numAttributes(); i++) {
			AttributeStats attrStats = dts.attributeStats(i);
			int missingCount = attrStats.missingCount;
			if (missingCount > 0) {
				hasMissing = true;
				double ratio = (double) missingCount / (double) numInst * 100;
				System.out.println("The missing attribute: " + dts.attribute(i).name() + " with ratio: " + ratio);
			}
		}
		if (!hasMissing)
			System.out.println("There is no missing value in this dataset");
	}

	static Instances getInstFromAttr(Instances dts, String[] colNames) {
		Instances copied = new Instances(dts);
		List<String> colList = Arrays.asList(colNames);
		for (int i = copied.numAttributes() - 1; i >= 0; i--) {
			String attrName = copied.attribute(i).name();
			if (!colList.contains(attrName)) {
				copied.deleteAttributeAt(i);
			}
		}
		return copied;
	}

	static Instances getInstExceptAttr(Instances dts, String[] colNames) {
		Instances copied = new Instances(dts);
		List<String> colList = Arrays.asList(colNames);
		for (int i = copied.numAttributes() - 1; i >= 0; i--) {
			String attrName = copied.attribute(i).name();
			if (colList.contains(attrName)) {
				copied.deleteAttributeAt(i);
			}
		}
		return copied;
	}

	static void standardScaler(Instances dts, String numColName) throws Exception {
		Attribute attr = dts.attribute(numColName);
		Instances filteredDts = getInstFromAttr(dts, new String[] { numColName });
		Standardize standardizeFilter = new Standardize();
		standardizeFilter.setInputFormat(filteredDts);
		filteredDts = Filter.useFilter(filteredDts, standardizeFilter);
		for (int i = 0; i < filteredDts.numInstances(); i++) {
			dts.instance(i).setValue(attr, filteredDts.instance(i).value(0));
		}
	}

	static double[] meanAndStd(Instances dts, String colName) {
		Attribute attr = dts.attribute(colName);
		double result = 0;
		if (!attr.isNumeric())
			return null;
		double mean = dts.meanOrMode(attr);

		int n = 0;
		for (int i = 0; i < dts.numInstances(); i++) {
			Instance inst = dts.instance(i);
			if (inst.isMissing(attr))
				continue;
			result += Math.pow(inst.value(attr) - mean, 2);
			n++;
		}
		double[] mAS = { mean, Math.sqrt(result / n) };
		return mAS;
	}

	static void meanFill(Instances dts, String numColName) throws Exception {
		Attribute attr = dts.attribute(numColName);
		if (!attr.isNumeric())
			throw new IllegalArgumentException("Input must be a column numeric type");
		double mean = meanAndStd(dts, numColName)[0];
		for (int i = 0; i < dts.numInstances(); i++) {
			Instance inst = dts.instance(i);
			if (inst.isMissing(attr)) {
				inst.setValue(attr, mean);
			}
		}
	}

	static String getLabel(double num, double[] bins, String[] labels) {
		if (num < bins[0] || num > bins[bins.length - 1]) {
			throw new IllegalArgumentException("Input number is out of range.");
		}
		for (int i = 0; i < bins.length - 1; i++) {
			if (num >= bins[i] && num < bins[i + 1]) {
				return labels[i];
			}
		}
		throw new IllegalArgumentException("Could not find label for input number.");
	}

	static HashMap<Object, Integer> binarize() {
		HashMap<Object, Integer> map = new HashMap<Object, Integer>();
		map.put(1, 1);
		map.put(0, 0);
		map.put(1.0, 1);
		map.put(0.0, 0);
		map.put("1", 1);
		map.put("0", 0);
		map.put("yes", 1);
		map.put("no", 0);
		map.put("true", 1);
		map.put("false", 0);
		return map;
	}

	static void le(Instances dts, String binaryColName) throws Exception {
		Instances delColInstances = getInstFromAttr(dts, new String[] { binaryColName });
		Attribute delAttribute = delColInstances.attribute(0);
		int delIndex = dts.attribute(binaryColName).index();
		int nunique = dts.attributeStats(delIndex).distinctCount;
		if (!(nunique == 2))
			throw new IllegalArgumentException("Input must be a binary columne!");
		dts.deleteAttributeAt(delIndex);
		Attribute newAttr = new Attribute(binaryColName, Arrays.asList("1", "0"));
		dts.insertAttributeAt(newAttr, delIndex);
		HashMap<Object, Integer> binarize = binarize();
		Integer __ = null;
		Instance inst = delColInstances.instance(0);
		Object val = null;
		Integer binary = null;
		if (delAttribute.isNumeric()) {
			val = inst.value(delAttribute);
		} else if (delAttribute.isString() || delAttribute.isNominal()) {
			val = inst.stringValue(delAttribute).toLowerCase();
		}
		binary = binarize.get(val);
		if (binary == null) {
			binary = 1;
			__ = 0;
		} else if (binary != null && binary == 1)
			__ = Integer.valueOf(0);
		else if (binary != null && binary == 0)
			__ = Integer.valueOf(1);
		if (delAttribute.isNumeric()) {
			for (int i = 0; i < delColInstances.numInstances(); i++) {
				Instance curr = delColInstances.instance(i);
				if ((double) val == curr.value(delAttribute)) {
					dts.instance(i).setValue(delIndex, __);
				} else
					dts.instance(i).setValue(delIndex, binary);
			}
		} else if (delAttribute.isString() || delAttribute.isNominal()) {
			for (int i = 0; i < delColInstances.numInstances(); i++) {
				Instance curr = delColInstances.instance(i);
				if (curr.stringValue(delAttribute).toLowerCase().equals((String) val)) {
					dts.instance(i).setValue(delIndex, __);
				} else {
					dts.instance(i).setValue(delIndex, binary);
				}
			}
		}
	}

	static void ohe(Instances dts, String categoryColName) throws Exception {
		Instances delColInstances = getInstFromAttr(dts, new String[] { categoryColName });
		Attribute delAttr = delColInstances.attribute(0);
		if (!delAttr.isNominal())
			throw new IllegalArgumentException("Input must be a nominal attribute type!");
		int delIndex = dts.attribute(categoryColName).index();
		dts.deleteAttributeAt(delIndex);
		HashMap<String, Integer> map = new HashMap<String, Integer>();
		Enumeration<Object> values = delAttr.enumerateValues();
		int colToAdd = delIndex;
		while (values.hasMoreElements()) {
			Object value = values.nextElement();
			Attribute newAttr = new Attribute(categoryColName + "_" + value, Arrays.asList("0", "1"));
			dts.insertAttributeAt(newAttr, colToAdd);
			map.put((String) value, colToAdd);
			colToAdd++;
		}
		int nAddedColumns = map.size();
		for (int i = 0; i < delColInstances.numInstances(); i++) {
			Instance inst = delColInstances.instance(i);
			boolean isMissing = inst.isMissing(delAttr);
			if (!isMissing) {
				int index = map.get(inst.stringValue(delAttr));
				for (int j = delIndex; j < delIndex + nAddedColumns; j++) {
					if (j == index)
						dts.instance(i).setValue(j, 1);
					else
						dts.instance(i).setValue(j, 0);
				}
			} else {
				for (int j = delIndex; j < delIndex + nAddedColumns; j++) {
					dts.instance(i).setMissing(j);
				}
			}
		}
		dts.deleteAttributeAt(colToAdd - 1);
	}

	static Instances[] trainTestSplit(Instances dts, int folds) throws Exception {
		if (dts.classIndex() == -1)
			throw new IllegalArgumentException("There is no class attribute");

		StratifiedRemoveFolds splitTest = new StratifiedRemoveFolds();
		splitTest.setNumFolds(folds);
		splitTest.setFold(1);
		splitTest.setSeed(1);
		splitTest.setInputFormat(dts);
		Instances testData = Filter.useFilter(dts, splitTest);

		StratifiedRemoveFolds splitTrain = new StratifiedRemoveFolds();
		splitTrain.setNumFolds(folds);
		splitTrain.setFold(1);
		splitTrain.setSeed(1);
		splitTrain.setInputFormat(dts);
		splitTrain.setInvertSelection(true);
		Instances trainData = Filter.useFilter(dts, splitTrain);

		return new Instances[] { trainData, testData };
	}

	static Instances smote(Instances dts, String classVal, double percent) throws Exception {
		SMOTE smote = new SMOTE();
		smote.setInputFormat(dts);
		smote.setClassValue(classVal);
		smote.setPercentage(percent);
		return Filter.useFilter(dts, smote);
	}

	static Instances oversampling(Instances dts) throws Exception {
		int classIndex = dts.classIndex();
		if (classIndex == -1)
			throw new IllegalArgumentException("There is no class attribute");
		Enumeration<Object> classes = dts.classAttribute().enumerateValues();
		HashMap<String, Integer> classesAppearance = new HashMap<String, Integer>();

		while (classes.hasMoreElements()) {
			String ele = (String) classes.nextElement();
			classesAppearance.put(ele, 0);
		}
		for (int i = 0; i < dts.numInstances(); i++) {
			String val = dts.instance(i).stringValue(classIndex);
			classesAppearance.put(val, classesAppearance.get(val) + 1);
		}
		int maxAppearance = 0;
		for (int value : classesAppearance.values()) {
			if (value > maxAppearance) {
				maxAppearance = value;
			}
		}
		final int finalMaxAppearance = maxAppearance;

		Enumeration<Object> classes2 = dts.classAttribute().enumerateValues();
		while (classes2.hasMoreElements()) {
			String classVal = (String) classes2.nextElement();
			int nappear = classesAppearance.get(classVal);
			if (nappear == finalMaxAppearance)
				continue;
			double percent = ((finalMaxAppearance - nappear) / (double) nappear) * 100;
			dts = smote(dts, classVal, percent);
		}

		return dts;
	}

	static <T extends AbstractClassifier> T buildClassifier(T csf, Instances dts, boolean cap) throws Exception {
		csf.buildClassifier(dts);
		if (cap) {
			System.out.println(
					"===========================================Capabilities===========================================");
			System.out.println(csf.getCapabilities().toString());
		}
		return csf;
	}

	static void printEvaluation(Evaluation eval) throws Exception {
		System.out.println(eval.toSummaryString());
		System.out.println(eval.toClassDetailsString());
		System.out.println(eval.toMatrixString());
		System.out.println("Area under ROC: " + eval.weightedAreaUnderROC());
	}

	static void holdoutEvaluateModel(AbstractClassifier csf, Instances trainDts, Instances testDts) throws Exception {
		Evaluation eval = new Evaluation(trainDts);
		buildClassifier(csf, trainDts, false);
		eval.evaluateModel(csf, testDts);
		String csfName = csf.getClass().getName();
		System.out.println("===========================================" + csfName
				+ "===========================================");
		printEvaluation(eval);
		System.out.println(
				"====================================================================================================");
	}

	static void kFoldsCrossValidationEvaluateModel(AbstractClassifier csf, Instances trainDts, int folds)
			throws Exception {
		Evaluation eval = new Evaluation(trainDts);
		buildClassifier(csf, trainDts, false);
		eval.crossValidateModel(csf, trainDts, folds, new Random(1));
		System.out.println("===========================================" + folds
				+ "-Folds Cross Validation Summary===========================================");
		printEvaluation(eval);
		System.out.println(
				"====================================================================================================");
	}

	static void kFoldsCrossValidationEvaluateModelDetails(AbstractClassifier csf, Instances trainDts, int folds)
			throws Exception {
		Instances randDts = new Instances(trainDts);
		randDts.randomize(new Random(1));
		if (randDts.classAttribute().isNominal())
			randDts.stratify(folds);
		System.out.println("===========================================" + folds
				+ "-Folds Cross Validation with " + csf.getClass().getName() + "===========================================");
		for (int i = 0; i < folds; i++) {
			Evaluation eval = new Evaluation(randDts);
			Instances train = randDts.trainCV(folds, i);
			Instances test = randDts.testCV(folds, i);
			buildClassifier(csf, train, false);
			eval.evaluateModel(csf, test);
			System.out.println("----------------------Fold: " + (i + 1) + "/" + folds + "----------------------");
			printEvaluation(eval);
		}
		kFoldsCrossValidationEvaluateModel(csf, trainDts, folds);
		System.out.println(
				"====================================================================================================");
	}

}
