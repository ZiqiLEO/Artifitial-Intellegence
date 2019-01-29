import numpy as np

def restrict(factor, variable, value):
    slices = []
    for i in range(factor.ndim):
	    if i == variable:
		    slices.append(value)
	    else:
		    slices.append(slice(None))
    return factor[slices]


def multiply(factor1, factor2, vars1, vars2, commonVars):
	f1 = []
	f2 = []
	
	for v in range(0, len(commonVars)):
		f1.append(1)
		f2.append(1)
	for v in range(0,len(commonVars)):
		if commonVars[v] in vars1:
			f1[v] +=1
		if commonVars[v] in vars2:
			f2[v] +=1

	shape1 = factor1.reshape(f1)
	shape2 = factor2.reshape(f2)
	result = shape1 * shape2
	return result
    
def sumout(factor,variable):
    result = factor.sum(axis = variable)
    return result

def normalize(factor):
    result = factor / factor.sum()
    return result

def findCommonVars(vars1, vars2, variables):
    commonVars = []
    for v in variables:
	    if v in vars1 or v in vars2:
		    commonVars.append(v)
    return commonVars
	    
	
def inference(factors, fac_vars, queryVariables, orderedHidVar, evidenceList, variables):
   
    #step1 Restrict factors in factorList according to evidence list
    #print(factors)
    #print(fac_vars)
    for f in factors:
	    Vars = fac_vars[f]
	    for v in evidenceList:
		    if v in Vars:
			    value = evidenceList[v]
			    variable = Vars.index(v)
			    #Restrict factors
			    newFactor = restrict(factors[f],variable,value)
			    
			    Vars.remove(v)
			    newVars = Vars
			    #newkey = max(factors,key=int) + 1
			    #add new factors to list
			    factors[f]= newFactor
			    fac_vars[f]=newVars
    #step2: sum out each hidden variable
    #print(factors)
    #print(fac_vars)
    for hidvar in orderedHidVar:
	    hidvar_factors = []
	    hidvar_factors_vars = []
	    remove_factors = []
	    for f in fac_vars:
		    if hidvar in fac_vars[f]:
			    hidvar_factors_vars.append(fac_vars[f])
			    hidvar_factors.append(factors[f])
			    remove_factors.append(f)
			    
	    
	    if(len(hidvar_factors) == 0):
		    continue
	    
	    product = hidvar_factors[0]
	    prod_vars = hidvar_factors_vars[0]
	    # multiply all factors contain hidden variable hid_var
	    for i in range (1, len(hidvar_factors)):
		    commonVars = findCommonVars(prod_vars, hidvar_factors_vars[i], variables)
		    
		    product = multiply(product, hidvar_factors[i],prod_vars, hidvar_factors_vars[i], commonVars)
		    #print("****multiplied factors*****")
		    #print(product)
		    #print("****************************")
		    prod_vars = commonVars
	    var_pos = prod_vars.index(hidvar)
	    # summing out the variable hid_var of the product 
	    product = sumout(product, var_pos)
	    
	    
	    prod_vars.pop(var_pos)
	    # put the product factor back to the factor list
	    	    
	    newkey = max(factors, key=int) + 1
	    factors.update({newkey: product})
	    fac_vars.update({newkey: prod_vars})
	    
	    for i in remove_factors:
		    fac_vars.pop(i)
		    factors.pop(i)	    
    
	    #print(factors)
	    #print(fac_vars)
	    #print("###################################")
    #step3: multiply the result factors to a single factor
    first = True
    #print(factors)
    for f in factors:
	    if first:
		    product = factors[f]
		    prod_vars = fac_vars[f]
		    first = False
	    else:
		    commonVars = findCommonVars(prod_vars, fac_vars[f], variables)
		    product = multiply(product, factors[f], prod_vars, fac_vars[f], commonVars)
		    #print("****multiplied factors*****")
		    #print(product)
		    #print("****************************")
		    prod_vars = commonVars
    #print(product)
		    
    #step 4: normalize
    result = normalize(product)
    #print(result)
    return result


def main():

    
	
	#PROBLEM 1(b,c,d,e)
	# f0(S)
	f0 = np.array([0.05, 0.95])	
		  
	# f1(M)
	f1 = np.array([0.03571, 0.96429])	
	
	#f2(NA)
	f2 = np.array([0.3,0.7])
	
	#f3(B,S)
	f3 = np.array([[0.6, 0.1],
	               [0.4, 0.9]])
	#f4(NH,M,NA)
	f4 = np.array([[[0.8,0.4],[0.5, 0]],
	               [[0.2,0.6],[0.5, 1]]])
	
	#f5(FH,S,NH,M)
	f5 = np.array([[[[0.99,0.75],[0.9,0.5]],
	                [[0.65,0.2],[0.4,0]]],
	              [[[0.01,0.25],[0.1,0.5]],
	               [[0.35,0.8],[0.6,1]]]])
	  
	variables = ["FH","B","S","NH","M","NA"] 			   
	
	
	# PROBLEM 1B	
	#factors dictionary
	factors = {0:f0, 1:f1, 2:f2, 3:f3, 4:f4, 5:f5}
	factorVars = {0:["S"], 1:["M"], 2:["NA"], 3:["B","S"],4:["NH","M","NA"], 5:["FH","S","NH","M"]}	
	evidence_list = {}
	quieryVar = "FH"
	prob_distribution = inference(factors, factorVars, quieryVar, ["B","M","NA","NH","S"], evidence_list, variables)
	print(prob_distribution)	
	
	
	
	# PROBLEM 1C	
	#factors dictionary
	factors = {0:f0, 1:f1, 2:f2, 3:f3, 4:f4, 5:f5}
	factorVars = {0:["S"], 1:["M"], 2:["NA"], 3:["B","S"],4:["NH","M","NA"], 5:["FH","S","NH","M"]}	
	evidence_list = {"M":0, "FH":0}
	quieryVar = "S"
	prob_distribution = inference(factors, factorVars, quieryVar, ["NA","NH","B"], evidence_list, variables)
	print(prob_distribution)	
	
	# PROBLEM 1D	
	#factors dictionary
	factors = {0:f0, 1:f1, 2:f2, 3:f3, 4:f4, 5:f5}
	factorVars = {0:["S"], 1:["M"], 2:["NA"], 3:["B","S"],4:["NH","M","NA"], 5:["FH","S","NH","M"]}	
	evidence_list = {"M":0, "FH":0, "B":0}
	quieryVar = "S"
	prob_distribution = inference(factors, factorVars, quieryVar, ["NA","NH"], evidence_list, variables)
	print(prob_distribution)	
	
	# PROBLEM 1E	
	#factors dictionary
	factors = {0:f0, 1:f1, 2:f2, 3:f3, 4:f4, 5:f5}
	factorVars = {0:["S"], 1:["M"], 2:["NA"], 3:["B","S"],4:["NH","M","NA"], 5:["FH","S","NH","M"]}	
	evidence_list = {"M":0, "FH":0, "B":0, "NA":0}
	quieryVar = "S"
	prob_distribution = inference(factors, factorVars, quieryVar, ["NH"], evidence_list, variables)
	print(prob_distribution)		
	
	
	# PROBLEM 2(a,b,c,d)
	# f0(B)
	f0 = np.array([0.1, 0.9])	
		  
	# f1(E)
	f1 = np.array([0.05, 0.95])	
	
	#f2(A,E,B)
	f2 = np.array([[[0.95, 0.1],[0.9,0.05]],
	               [[0.05,0.9],[0.1,0.95]]])
	
	#f3(W,A)
	f3 = np.array([[0.8, 0.4],
	               [0.2, 0.6]])
	#f4(G,A)
	f4 = np.array([[0.4,0.05],
	               [0.6,0.95]])
	variables = ["W","G","A","E","B"]
	'''
	'''
	# PROBLEM 2A	
	#factors dictionary
	factors = {0:f0, 1:f1, 2:f2, 3:f3, 4:f4}
	factorVars = {0:["B"], 1:["E"], 2:["A","E","B"], 3:["W","A"],4:["G","A"]}	
	evidence_list1 = {"W":1}
	quieryVar = "G"
	
	prob_distribution = inference(factors, factorVars, quieryVar, ["B","E","A"], evidence_list1, variables)
	print(prob_distribution)
	
	#evidence_list2 = {"W":0}
	#prob_distribution1 = inference(factors, factorVars, quieryVar, ["B","E","A"], evidence_list2, variables)
	#print(prob_distribution1)	
    
    
	# PROBLEM 2B	
	#factors dictionary
	factors = {0:f0, 1:f1, 2:f2, 3:f3, 4:f4}
	factorVars = {0:["B"], 1:["E"], 2:["A","E","B"], 3:["W","A"],4:["G","A"]}	
	evidence_list1 = {"W":0, "G":0, "A":0}
	quieryVar = "B"
	
	#prob_distribution = inference(factors, factorVars, quieryVar, ["E"], evidence_list1, variables)
	#print(prob_distribution)
	
	evidence_list2 = {"A":0}
	prob_distribution1 = inference(factors, factorVars, quieryVar, ["E","G","W"], evidence_list2, variables)
	print(prob_distribution1)
	
    
	# PROBLEM 2C	
	#factors dictionary
	factors = {0:f0, 1:f1, 2:f2, 3:f3, 4:f4}
	factorVars = {0:["B"], 1:["E"], 2:["A","E","B"], 3:["W","A"],4:["G","A"]}	
	evidence_list1 = {"A":0, "G":0, "W":0}
	quieryVar = "B"
	
	#prob_distribution = inference(factors, factorVars, quieryVar, ["E"], evidence_list1, variables)
	#print(prob_distribution)
	
	evidence_list2 = {"W":0}
	prob_distribution1 = inference(factors, factorVars, quieryVar, ["E","A","G"], evidence_list2, variables)
	print(prob_distribution1)
	
    
	# PROBLEM 2D	
	#factors dictionary
	factors = {0:f0, 1:f1, 2:f2, 3:f3, 4:f4}
	factorVars = {0:["B"], 1:["E"], 2:["A","E","B"], 3:["W","A"],4:["G","A"]}	
	evidence_list1 = {"A":0, "B":0}
	quieryVar = "E"
	
	#prob_distribution = inference(factors, factorVars, quieryVar, ["G","W"], evidence_list1, variables)
	#print(prob_distribution)
	
	evidence_list2 = {"A":0}
	prob_distribution1 = inference(factors, factorVars, quieryVar, ["B","G","W"], evidence_list2, variables)
	print(prob_distribution1)	
	
	
	
	
	
if __name__ == "__main__":
    main()


	    
	    
		    
			    
		
			
			    
	    
	
    
    
    


