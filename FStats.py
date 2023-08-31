import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb
import math
from scipy.stats import norm
import scipy

class Stats:
    
    def __init__(self, data):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        self.df = data
        
    def fact(self,i):
        x = math.factorial(i)
        return x
    def comb(self,n,r):
        x = math.comb(n,r)
        return x
    def perm(self,n,r):
        x = math.perm(n,r)
        return x
    def get_µX(self,n,p):
        µ= n*p
        return µ
    def get_σX(self,n,p):
        σ = math.sqrt((n*p*(1-p)))
        return σ
    def get_µX_and_σX(self,n,p):
        µ=self.get_µX(n,p)
        σ=self.get_σX(n,p)
        return µ,σ
    def get_σ_X̅(self,σ,n):
        X̅=σ/math.sqrt(n)
        return X̅
    def get_σ_p_hat(self,n,p):
        σ_p = math.sqrt((p*(1-p)/n))
        return σ_p
    def get_z_0(self,p_0,p_hat,n):
        z_0 = (p_hat-p_0)/math.sqrt((p_0*(1-p_0)/n))
        return z_0
    def get_confidence_interval_norm(self,x,n,C):
        p_hat = x/n
        check = n*p_hat*(1-p_hat)
        if check  >= 10:
            print('data is normally distributed: ',check)
        else:
            print('WARNING: data is NOT normally distributed WARNING!!! : ',check)
        l,h = norm.interval(C)
        lower_bound = p_hat + l * math.sqrt(p_hat*(1-p_hat)/n)
        upper_bound = p_hat + h * math.sqrt(p_hat*(1-p_hat)/n)
        return lower_bound, upper_bound
    
    def get_population_estimate_for_n(self,C,E,p_hat=None):
        za=1-C
        if p_hat == None:
            n=0.25*((norm.ppf(1-round(za,8)/2))/E)**2
        else:
            n=p_hat*(1-p_hat)*((norm.ppf(1-round(za,6)/2))/E)**2
        return n
    
    def get_t_value(self,q,dof,right=None,two_tailed=False):
        t=0
        if right == False:
            t=scipy.stats.t.ppf(q=q,df=dof)
        if right == True:
            t=scipy.stats.t.ppf(q=1-q,df=dof)
        if two_tailed ==True:
            t=scipy.stats.t.ppf(q=1-(1-q)/2,df=dof)
        return t

    def get_t_confidence_interval(self,x_bar, s_std, t_value,n):
        lb = x_bar - t_value*(s_std/math.sqrt(n))
        ub = x_bar + t_value*(s_std/math.sqrt(n))
        return (lb,ub)

    def get_sample_size_estimate(self,E,s,za):
        n = math.ceil(((norm.ppf(za))*s/E)**2)
        return n

    def get_proportion_confidence_interval(self,n,p,za):
        lb=p-za*math.sqrt((p*(1-p))/n)
        ub=p+za*math.sqrt((p*(1-p))/n)
        return (lb,ub)
    def get_t_distribution(self,x_bar,µ,s,n):
        t = (x_bar-µ)/(s/math.sqrt(n))
        return t

    def get_p_value_from_t_stat(self,t,n,two_tailed=False):
        if two_tailed==True:
            p_value=scipy.stats.t.sf(abs(t), df=(n-1))*2
        else:
            p_value=scipy.stats.t.sf(abs(t), df=(n-1))
        return p_value

        
    def big13 (self, ddof,col_name,p=0):
        '''Self:  DataFrame
        DDOF:    Degrees of Delta Freedom
        p:       Prints count,STD,mean,min/max values, range, Q1,Q2,Q3,IQR,
                 Upper/Lower Fence, and varriance
        
        returns count,mean,std,minV,Q1,Q2,Q3,maxV,vari
        '''
        count = self.df[col_name].count()
        std=0
        mean = self.df[col_name].mean()
        minV= self.df[col_name].min()
        maxV=self.df[col_name].max()
        rangeV = maxV - minV
        Q1,Q2,Q3 = self.df[col_name].quantile([.25,.5,.75])
        IQR = Q3-Q1
        UL = Q3 + (IQR*1.5)
        LL = Q1 - (IQR*1.5)
        if ddof == 0:
            std = self.df[col_name].std(ddof=0)
        if ddof == 1:
            std = self.df[col_name].std(ddof=1)
        n = len(self.df[col_name])
        mean1 = sum(self.df[col_name]) / n
        vari= sum((x - mean1) ** 2 for x in self.df[col_name]) / (n - ddof)
        if p !=0:

            print('count ',count,'\nMean ',mean,"\nSTD ",std,'\nLower Limit  ',LL,"\nMinimum ",minV,'\nQ1 ',Q1,'\nQ2 ',Q2,'\nQ3 ',
                  Q3,'\nMaximum ',maxV,'\nRange ',rangeV,"\nIQR ",IQR,'\nVariance: ',vari,'\nUpper Limit ',UL)
        return count,mean,std,minV,Q1,Q2,Q3,maxV,vari
    def bin_app(self,range_val,frequency,ddof=1):
        '''DDOF is Delta Degrees of Freedom
        Self: needs to be a DataFrame with range and f for columns
        - also needs an extra Lower limit bin with a frequency of zero 
          added to the bottom of the list
        returns: Dataframe, Approximate Mean, and Approx STD
        '''
        self.df['Xi']=0
        self.df['Xi,Fi'] = 0
        self.df['Xi-Xbar'] = 0
        self.df['(Xi-Xbar)squared*Fi'] = 0
        i=0
        i3=len(self.df[frequency])-1

        for f in self.df[frequency]:
            i2 = i + 1
            if i < i3 :
                r = self.df[range_val].iloc[i]
                r2 = self.df[range_val].iloc[i2]
                Xi = (r+r2)/2
                self.df.loc[i,'Xi'] = Xi
                Fi = Xi*f
                self.df.loc[i,'Xi,Fi']= Fi
                i = i+1
            else:
                pass

        Fi_sum = self.df[frequency].sum()
        XiFi_sum = self.df['Xi,Fi'].sum()
        app_mean = XiFi_sum/Fi_sum
        print("app Mean: ",app_mean)
        i=0
        for x in self.df['Xi']:
            if i < i3:
                dev = x-app_mean
                self.df.loc[i,'Xi-Xbar']=dev
                i=i+1
        i= 0
        for y in self.df['Xi-Xbar']:
            z = y*y*self.df[frequency].iloc[i]
            self.df.loc[i,'(Xi-Xbar)squared*Fi']=z
            i = i+1
        dev_sum2 = self.df['(Xi-Xbar)squared*Fi'].sum() 
        f_sum_adj = self.df[frequency].sum()-ddof
        squared_std= dev_sum2/f_sum_adj
        app_std = math.sqrt(squared_std)
        print('app std:  ', app_std)
        return self.df, app_mean, app_std
    
    def bin_freq(self,bins,col_name):
        '''Self:   Pandas Dataframe
        col_name:  Column in Dataframe
        bins:      [] bins have to be bigger than largest number
        Returns:   New dataframe with frequency
        '''
        
        new_df = pd.DataFrame(bins,columns=['Range'])
        new_df['Frequency']=0
        new_df.set_index('Range')
        i = 0
        for r in bins:
            i2 = i + 1
            for v in self.df[col_name]:
                if v >= new_df['Range'].iloc[i]:
                    if i2 <= len(new_df['Range']):
                        if v < new_df['Range'].iloc[i2]:
                            new_df['Frequency'].iloc[i] += 1
            i = i + 1
        return new_df
    
    def sample_combo(self,col_name,n=0,u=0,r=0):
        '''self:  DataFrame
        col_name: name of column on Dataframe
        n:        number of sample combinations
        u:        Unique=1
        r:        Combinations with replacement=1
        '''
        if u == 1:
            self.df[col_name] = set(self.df[col_name])
        LS_combo = list()
        combo=[]
        if r == 1:
             for i in range(len(self.df[col_name]) +1):
                LS_combo += list(math.perm(self.df[col_name], i))
        else:
            for i in range(len(self.df[col_name]) +1):
                LS_combo += list(math.comb(self.df[col_name], i))
        if n != 0:
            for i in LS_combo:
                if len(i) == n:
                    combo.append(i)
        else:
            combo = LS_combo
        return combo
    def empirical_rule(self,mean,std):
        '''Input Mean and STD manually to use this function'''
        v65u = mean+std
        v65l = mean-std
        v95u = mean+std*2
        v95l = mean-std*2
        v99_7u= mean+std*3
        v99_7l= mean-std*3
        print('65% of data is between: ',v65l," and: ",v65u,
              '\n95% of data is between: ',v95l,' and: ',v95u,
              '\n99.7% of data is bewtwwen: ',v99_7l,' and: ',v99_7u,
              '\n\nSTD-3(.15): ',v99_7l,'----------------------------------------------|.15^',
              '\n | .0235 |                                                         --|',
              '\nSTD-2(2.3): ',v95l,'--------------------------------|                  --|',
              "\n | .135 |                                       --|                     --|",
              '\nSTD-1(15.9): ',v65l,'---------------|               --|                      --|',
              '\n | .34 |                         --|              --|                         --|',
              '\n50% of data is under: ', mean,'        65%              95%                       97.7%'
              '\n | .34 |                         --|              --|                         --|',
              '\nSTD+1(84.1): ',v65u,'---------------|               --|                      --|',
              '\n | .135 |                                       --|                     --|',
              '\nSTD+2(97.7): ',v95u,'------------------------------|                   --|',
              '\n | .0235 |                                                         --|',
              '\nSTD+3(99.9): ',v99_7u,'----------------------------------------------|.15v')
    def freq_counter(self,col_name):
        '''self:   DataFrame
        col_name:  Name of the column in dataframe
        '''
        frequency= pd.DataFrame(columns=['frequency','relative frequency'])
        counter = 0
        for i in self.df[col_name]:
            counter = counter+1
            if i in frequency['frequency']:
                frequency.loc[i,'frequency'] += 1
            else:
                frequency.loc[i,'frequency'] = 1
        i=0
        for value in frequency['frequency']:
            frequency['relative frequency'].iloc[i] = value/counter
            i=i+1
        return frequency
    
    def get_z_score(self,mean,std,x):
        '''Input Mean, STD, and the value for x to be examined
        '''
        z = (x-mean)/std
        print('Z-Score: ',z)
        return z
    
    def histogram(self,col_name,bins,kde=False):
        '''Self:   DataFrame
        col_name:  Name of column in dataframe
        bins:      [] of bins
        '''
        ax = self.df[col_name].plot(kind='hist',bins=bins)
        if kde == True:
            self.df[col_name].plot(kind='kde',ax=ax,secondary_y=True)
        plt.show()
    def suppress(self,number):
        '''Self: DataFrame
        number:  Number that you want to suppres from scientific notation'''
        n='{:f}'.format(number)
        return n
    def boxplot(self,col_name):
        '''Self:   Dataframe
        col_name:  Name of column from dataframe
        '''
        boxplot=self.df.boxplot(col_name)
        boxplot.plot()
        plt.show()
    def barchart(self,col_name,index,i=0):
        '''Self:  DataFrame
        col_name: Name of column wanting charted
        index:    Only use if Index is not set- Column name for index
        i:        Change to 1 if needing to set index
        '''
        if i == 1:
            self.df.set_index(index,inplace=True)
        fig, ax = plt.subplots()
        self.df[col_name].plot(ax=ax, kind='bar')
    
    def scatterplot(self,x,y,c='blue'):
        x=self.df[x]
        y=self.df[y]
        plt.scatter(x,y,c=c)
        
    def Pearson_corr_coeff(self,x_col_name,y_col_name, numeric_only=False):
        '''Self: DataFrame
        x_col_name: First variable column
        y_col_name: Second variable column
        
        '''
        correlation = self.df.corr(method='pearson',numeric_only=numeric_only)
        r=correlation.loc[x_col_name,y_col_name]
        r2 = round(r,8)
        R = r2*r2*100
        R2=round(R,8)
        print('Is correlation coefficient close to 0?    r:'+str(r2))
        print('Coefficient of determination is  R: '+str(R2)+'%')
        return r2,R2
    
    def slope(self,P,Q):
        '''P and Q are Tuples'''
        s1 = (Q[1]-P[1])  
        s2 = (Q[0]-P[0])
        m = s1/s2
        m1 = str(s1) + '/'+ str(s2) + '(x-'
        print('y-'+str(P[1])+'='+m1 + str(P[0])+')')
        return m

    def x_and_y_intercept(self,P, Q):
        ''' TAKES A TUPLE for P and Q! 
        Function to find the X and Y intercepts
     of the line passing through
     the given points'''
        a = P[1] - Q[1]
        b = P[0] - Q[0]
        # if line is parallel to y axis
        if b == 0:
            print(P[0])         # x - intercept will be p[0]
            print("infinity")   # y - intercept will be infinity
            return
        # if line is parallel to x axis
        if a == 0:
            print("infinity")     # x - intercept will be infinity
            print(P[1])           # y - intercept will be p[1]
            return
        # Slope of the line
        m = a / b
        # y = mx + c in where c is unknown
        # Use any of the given point to find c
        x = P[0]
        y = P[1]
        c = y-m * x
        # For finding the x-intercept put y = 0
        y = 0
        x =(y-c)/m
        print('x-intercept:' + str(x))  
        # For finding the y-intercept put x = 0
        x1 = 0
        y = m * x1+ c
        print('y-intercept:' + str(y))
        return x,y
        
    def least_squares_regression(self,x_col_name,y_col_name,ddof=1,dot_size=100,x=None,slope2=None,intercept2=None,r=0):
        '''self: DataFrame
        x_col_name: Name of x axis column in dataframe
        y_col_name: name of y axis column in dataframe
        ddof: Delta Degrees of freedom
        dot_size: int for scatterplot dot size
        x: predicted value of least squares regression
        slope2: int or float of second line to plot
        intercept2: int or float of second line to plot
        r: if linear correlation coefficient is provided
        '''
        def abline(slope, intercept,l='--',c='blue'):
            '''slope: int or float
            intercept: int or float
            slope2: int or float of a second line to plot
            intercept2: int or float of second line to plot'''
            axes = plt.gca()
            x_vals = np.array(axes.get_xlim())
            y_vals = intercept + slope * x_vals
            plt.plot(x_vals, y_vals, l,c=c)
        R=0
        if r==0:
            r,R = self.Pearson_corr_coeff(x_col_name,y_col_name)
            #print('r= '+str(r))
            
        xcount,xmean,xstd,xminV,xQ1,xQ2,xQ3,xmaxV,xvari = self.big13(ddof,x_col_name)
        ycount,ymean,ystd,yminV,yQ1,yQ2,yQ3,ymaxV,yvari = self.big13(ddof,y_col_name)
        b1 = r*(ystd/xstd)
        #print('b1= '+str(b1))
        b0 = ymean - (b1*xmean)
       # print('b0= '+str(b0))
        yhat= 'yhat='+str(b1)+'x+'+str(b0)
        print(yhat)
        
        self.df.plot.scatter(x=x_col_name,y=y_col_name,s=dot_size,grid=True)
        line = abline(b1,b0)
        self.df['Y Prediction'] = 0
        self.df['Residual^2'] = 0
        self.df['Residual'] = 0
        i=0
        for x1 in self.df[x_col_name]:
            y1 = (b1*x1)+b0
            self.df.iloc[i,'Y Prediction'] = y1
            self.df.loc[i,'Residual'] =(self.df.loc[i,y_col_name]-y1)
            self.df.loc[i,'Residual^2'] = (self.df.loc[i,y_col_name]-y1)*(self.df.loc[i,y_col_name]-y1)
            i = i+1
        
        if x != None:
            y = (b1*x)+b0
            print('pridicted value is: '+str(y))
            
        if slope2 != None:
            line2 = abline(slope2,intercept2,c='red',l='-')
            i=0
            self.df['Y Prediction B'] = 0
            self.df['Residual^2 B'] = 0
            self.df['Residual B'] = 0
            for x1 in self.df[x_col_name]:
                y1 = (slope2*x1)+intercept2
                self.df.loc[i,'Y Prediction B'] = y1
                self.df.loc[i,'Residual B'] =(self.df.loc[i,y_col_name]-y1)
                self.df.loc[i,'Residual^2 B'] = (self.df.loc[i,y_col_name]-y1)*(self.df.loc[i,y_col_name]-y1)
                i = i+1

        plt.show()
        return b1,b0
    def farey(self,x, N):
        '''convert a decimal to a fraction
        x:   is decimal to be converted between -1 and 1
        N:    is max denom'''
        a, b = 0, 1
        c, d = 1, 1
        while (b <= N and d <= N):
            mediant = float(a+c)/(b+d)
            if x == mediant:
                if b + d <= N:
                    return a+c, b+d
                elif d > b:
                    return c, d
                else:
                    return a, b
            elif x > mediant:
                a, b = a+c, b+d
            else:
                c, d = a+c, b+d

        if (b > N):
            return c, d
        else:
            return a, b
    def critical_values_corr_coeff(self):
        cvcc={3:0.997,4:0.950,5:0.878,6:0.811,7:0.754,8:0.707,9:0.666,10:0.632,11:0.602,12:0.576,13:0.553,14:0.532,15:0.514,
         16:0.497,17:0.482,18:0.468,19:0.456,20:0.444}
        return cvcc
            
    def probability_emperical(self,col_name):
        p_str = str(col_name) + ' P'
        self.df[p_str] = (self.df[col_name]/self.df[col_name].sum())
        
    def sample_space(self, col_name,ns):
        #sample space with replacement
        r = list(product(self.df[col_name],repeat=ns))
        return r
    def binomial_probability_distribution(self,n,p,x):
        r = self.comb(n,x)*(p**x)*(1-p)**(n-x)
        return r
    def cumulative_binomial_probability_distribution(self,n,p,x):
        total = 0
        for i in range(0,x,1):
            r = self.comb(n,i)*(p**i)*(1-p)**(n-i)
            total = total + r
        return total
    def range_binomial_probability_dist(self,n,p,x):
        df = pd.DataFrame(x,columns=['x'])
        df['P(x)']=0
        def binomial_probability_distribution(n,p,x):
            r = self.comb(n,x)*(p**x)*(1-p)**(n-x)
            return r
        for i in x:
            df.loc[i,'P(x)'] = binomial_probability_distribution(n,p,i)
        return df
    def mean_std_discrete_randvar(self,x_col,Px_col):
        self.df['x*P(x)']=self.df[x_col]*self.df[Px_col]
        µ = self.df['x*P(x)'].sum()
        print('mean: ',µ)
        self.df['x²·P(x)-µ²'] = ((self.df[x_col]**2)*self.df[Px_col])
        var=(self.df['x²·P(x)-µ²'].sum()-µ**2)
        σ = math.sqrt(var)
        print('std: ',σ)
        return µ,σ
    
    def kde_plot(self,col_name,greater_than=False,val=None,out=None,between=None):
        
        µ=self.df[col_name].mean()
        std=self.df[col_name].std()
        σ1 = µ + std
        σ_1= µ - std
        σ2 = µ + std*2
        σ_2= µ - std*2
        σ3 = µ + std*3
        σ_3= µ - std*3
        
        sb.set_style('whitegrid')
        plt.figure()
        sb.kdeplot(self.df[col_name],linewidth=2,fill=True)
        ax = sb.kdeplot(self.df[col_name])
        trans = ax.get_xaxis_transform()
        kde_lines = ax.get_lines()[-1]
        kde_x, kde_y = kde_lines.get_data()
        
        ax.axvline(σ1,0,.60,color='black',linewidth=.9, linestyle='--',)
        ax.annotate('σ1',xy=(σ1, 0),fontsize=10)
        ax.text(σ1, 0.6,'<--34%',fontsize=8,transform=trans)
        ax.text(σ1, 0.65,'84%C',fontsize=8,transform=trans,color='red')
        
        ax.axvline(σ2,0,.13,color='black',linewidth=.7, linestyle='--')
        ax.annotate('σ2',xy=(σ2, 0),fontsize=10)
        ax.text(σ2, 0.13,'<--13.5%',fontsize=8,transform=trans)
        ax.text(σ2, 0.18,'97.5%C',fontsize=8,transform=trans,color='red')
        
        ax.axvline(σ3,0,.02,color='black',linewidth=.5, linestyle='--')
        ax.annotate('σ3',xy=(σ3, 0),fontsize=10)
        ax.text(σ3, 0.05,'<--2.35%  0.15%-->',fontsize=8,transform=trans)
        ax.text(σ3, 0.1,'99.85%C',fontsize=8,transform=trans,color='red')
        
        ax.axvline(µ,0,.95,color='black',linewidth=1.6,)
        ax.annotate('µ',xy=(µ, 0),fontsize=12)
        ax.text(µ, 0.98,'50%',fontsize=10,transform=trans,color='red')
        
    
        ax.axvline(σ_1,0,.60,color='black',linewidth=.9, linestyle='--')
        ax.annotate('σ-1',xy=(σ_1, 0),fontsize=10,horizontalalignment='left')
        pl1 = σ_1-((kde_x.max()-kde_x.min())*0.05)
        ax.text(pl1, 0.6,'34%',fontsize=8,transform=trans)
        ax.text(pl1, 0.65,'16%C',fontsize=8,transform=trans,color='red')
        
        ax.axvline(σ_2,0,.13,color='black',linewidth=.7, linestyle='--')
        ax.annotate('σ-2',xy=(σ_2, 0),fontsize=10,horizontalalignment='left')
        pl2 = σ_2-((kde_x.max()-kde_x.min())*0.05)
        ax.text(pl2, 0.13,'13.5%',fontsize=8,transform=trans)
        ax.text(pl2, 0.18,'2.5%C',fontsize=8,transform=trans,color='red')
        
        ax.axvline(σ_3,0,.02,color='black',linewidth=.5, linestyle='--')
        ax.annotate('σ-3',xy=(σ_3, 0),fontsize=8,horizontalalignment='left')
        pl3 = σ_3-((kde_x.max()-kde_x.min())*0.16)
        ax.text(pl3, 0.05,'<--0.15% 2.35%-->',fontsize=8,transform=trans)
        ax.text(pl3, 0.1,'0.15%C',fontsize=8,transform=trans,color='red')
        
        if val != None:
            if greater_than == True:
                mask = (kde_x > val)
                filled_x, filled_y = kde_x[mask],kde_y[mask]
                ax.fill_between(filled_x,y1=filled_y,color='red')
            else:
                mask = (kde_x < val)
                filled_x, filled_y = kde_x[mask],kde_y[mask]
                ax.fill_between(filled_x,y1=filled_y,color='red')

        if out != None:
            mask = (kde_x < out[0])
            filled_x, filled_y = kde_x[mask],kde_y[mask]
            ax.fill_between(filled_x,y1=filled_y,color='red')

            mask = (kde_x > out[1])
            filled_x, filled_y = kde_x[mask],kde_y[mask]
            ax.fill_between(filled_x,y1=filled_y,color='red')
        if between != None:
            mask = (kde_x > between[0]) & (kde_x <between[1] )
            filled_x, filled_y = kde_x[mask],kde_y[mask]
            ax.fill_between(filled_x,y1=filled_y,color='red')
        plt.show()
        
    def normal_distribution_generator(self,mu,std,size):
        randnum = np.random.normal(loc=mu,scale=std,size=size)
        df= pd.DataFrame(randnum,columns=['x'])
        return df
