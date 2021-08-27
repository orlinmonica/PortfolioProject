use JaponesePigure

--no1
select CustomerName 
from dbo.Customer
where CustomerId IN (select CustomerId 
		from dbo.HeaderTransaction where DeliveryType='Tudey')
order by CustomerName ASC

--no2
select C.CustomerName, W.WorkerName
from dbo.HeaderTransaction H
join dbo.Customer C
on H.CustomerId = C.CustomerId
join dbo.Worker W
on H.WorkerId = W.WorkerId
where W.WorkerSalary > (select avg(WorkerSalary) from dbo.Worker)
	AND W.WorkerGender = 'Female'


select year(TransactionDate) from dbo.HeaderTransaction
--no3
select W.WorkerName, count(W.WorkerId) as 'Total Transaction'
from dbo.HeaderTransaction H 
join dbo.Worker W
on H.WorkerId = W.WorkerId
join dbo.DetailTransaction D
on H.TransactionId = D.TransactionId
where W.WorkerGender = 'Female' 
	AND Year(H.TransactionDate) = 2019
group by W.WorkerName

--no.4
select top 3 C.CustomerName, count(C.CustomerId) as 'Total Transaction'
from dbo.Customer C
join dbo.HeaderTransaction H
on C.CustomerId = H.CustomerId
join dbo.DetailTransaction D
on H.TransactionId = D.TransactionId
join dbo.Figurine F
on D.FigurineId = F.FigurineId
where F.FigurinePrice > 500000
	AND F.FigurineGrade = 'Normal'
group by C.CustomerName
order by 'Total Transaction' DESC


select top 3 C.CustomerName, count(C.CustomerId) as 'Total Transaction'
from dbo.Customer C
join dbo.HeaderTransaction H
on C.CustomerId = H.CustomerId
join dbo.DetailTransaction D
on H.TransactionId = D.TransactionId
join dbo.Figurine F
on D.FigurineId = F.FigurineId
where F.FigurineId IN
	(select FigurineId
	from dbo.Figurine
	where FigurinePrice > 500000
	AND FigurineGrade = 'Normal')
group by C.CustomerName
order by 'Total Transaction' DESC

select C.CustomerName, count(C.CustomerId) as 'Total Transaction' 
from dbo.Customer C
 left join dbo.HeaderTransaction H
on H.CustomerId = C.CustomerId
left join dbo.DetailTransaction D
on D.TransactionId = H.TransactionId
group by C.CustomerName


--no5
select C.CustomerName , F.FigurineName
from dbo.Customer C
join dbo.HeaderTransaction H
on C.CustomerId = H.CustomerId
join dbo.DetailTransaction D
on H.TransactionId = D.TransactionId
join dbo.Figurine F
on D.FigurineId = F.FigurineId
where year(H.TransactionDate) > 2018
	AND len(C.CustomerName) > 15

--no6
select distinct substring(C.CustomerId, 3,3) as 'ID', 
	SUBSTRING(C.CustomerName,CHARINDEX(' ',C.CustomerName),100) as 'Given Name'
from dbo.Customer C
join dbo.HeaderTransaction H
on H.CustomerId=C.CustomerId
join dbo.Worker W
on W.WorkerId=H.WorkerId
where W.OriginCity in
	(select OriginCity
	from dbo.Worker
	where OriginCity ='Jakarta' 
	OR OriginCity = 'Sukabumi')
order by 'ID' 


select SUBSTRING(CustomerName,CHARINDEX(' ',CustomerName),100)
from dbo.Customer
select CHARINDEX(' ',CustomerName) from dbo.Customer

--no7
select distinct WorkerName
from dbo.Worker W
join dbo.HeaderTransaction H
on H.WorkerId = W.WorkerId
where W.WorkerId IN
	(select WorkerId 
	from dbo.HeaderTransaction
	where (DeliveryType='Tudey'
		OR DeliveryType='Neks Dey')
		AND PaymentType ='Kreditz')
select * from dbo.HeaderTransaction

--no.8
select * from dbo.Figurine
select FigurineGrade, count(FigurineGrade) AS 'Number Of Available Figurine'
from dbo.Figurine
where FigurinePrice > 
	all(select AVG(FigurinePrice)
	from dbo.Figurine)
group by FigurineGrade

--no.9
--cara gapake alias subquery
select f1.FigurineName, f1.FigurineGrade
from 
	(select FigurineName, FigurineGrade
	from dbo.Figurine
	where (FigurinePrice in
		(select max(FigurinePrice)
		from dbo.Figurine))  OR
		 (FigurinePrice in
		(select min(FigurinePrice)
		from dbo.Figurine))) f1

--cara lins pake join tpi failed
select f1.FigurineName, f1.FigurineGrade
from dbo.Figurine f1
join (select FigurineId, max(FigurinePrice) as 'maxprice'
	from dbo.Figurine
	where FigurinePrice = 
	(select max(FigurinePrice)
	from dbo.Figurine)
	group by FigurineId) f2 
on f1.FigurineId = f2.FigurineId
full outer join (select FigurineId, min(FigurinePrice) as 'minprice'
	from dbo.Figurine
	where FigurinePrice = 
	(select min(FigurinePrice)
	from dbo.Figurine)
	group by FigurineId) f3
on f1.FigurineId = f3.FigurineId
where (f1.FigurinePrice = f2.maxprice) OR
	(f1.FigurinePrice = f3.minprice)

--cara capi
select res.FigurineName, res.FigurineGrade
from 
( (select FigurineName, FigurineGrade 
   from dbo.figurine
    where FigurinePrice = (select max(FigurinePrice) from dbo.figurine) ) 
    UNION
    (select FigurineName, FigurineGrade from dbo.figurine
where FigurinePrice = (select min(FigurinePrice)
    from dbo.figurine) ) )  res

--no10
select sub.TSY as 'Top Selling Year'
from (
	select year(TransactionDate) as TSY
	from dbo.HeaderTransaction H
	join dbo.DetailTransaction D
	on H.TransactionId = D.TransactionId
	join dbo.Figurine F
	on F.FigurineId = D.FigurineId
	where F.FigurinePrice = 
		(select max(FigurinePrice) 
		from dbo.Figurine)) sub
